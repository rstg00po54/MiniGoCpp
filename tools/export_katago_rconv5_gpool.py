#!/usr/bin/env python3
"""Export actual KataGo 10b rconv5 global-pooling + w1r ONNX CPU golden.

An isolated, unchanged upstream opset19/IR9 graph produces real rconv5 pool
input, mask scale, and w1r output. Full InputMask is all ones, so the masked
max source equals the pooled input and mean reduces to spatial mean.

Packet K10GP001, little endian: 8-byte magic, four uint32 H W Cin Cout,
one FP16 scalar gpool-scale, FP16 NHWC pooled input, FP16 OHWI 1x1 w1r weights,
FP32 NHWC w1r reference. All extracted from the original KataGo 10b ONNX.
"""
import argparse
import copy
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def export(model_path, out_path, seed=42):
    model = onnx.load(model_path, load_external_data=True)
    onnx.checker.check_model(model)
    g = model.graph
    nodes = {name:i for i,n in enumerate(g.node) for name in n.output if name}
    initial = {x.name:x for x in g.initializer}
    inputs = {x.name:x for x in g.input}
    source = "rconv5/gpool/mask/70"
    scale_name = "InputMask/scale/6"
    output = "rconv5/w1r/79"
    if any(x not in nodes for x in (source, scale_name, output)):
        raise ValueError("Expected 10b rconv5 gpool tensors were not found")
    wn = next((n for n in g.node if n.name=="rconv5/w1r"),None)
    if wn is None or wn.op_type!="Conv" or len(wn.input)<2:
        raise ValueError("rconv5/w1r must be a Conv with constant weights")
    if wn.input[1] not in initial:
        raise ValueError("rconv5/w1r weights are not constant")
    weights=np.asarray(numpy_helper.to_array(initial[wn.input[1]]),dtype=np.float32)
    if weights.shape!=(96,96,1,1):
        raise ValueError("Expected 96x96 1x1 gpool Conv weights: "+str(weights.shape))
    attrs={a.name:helper.get_attribute_value(a) for a in wn.attribute}
    if (list(attrs.get("strides",[1,1]))!=[1,1] or
        list(attrs.get("pads",[0,0,0,0]))!=[0,0,0,0] or
        int(attrs.get("group",1))!=1):
        raise ValueError("Unexpected rconv5/w1r attrs: "+str(attrs))
    if len(wn.input)>2 and wn.input[2]:
        raise ValueError("Unexpected w1r bias operand; packet does not encode bias")

    required=set()
    def walk(v):
        if v in initial or v in inputs: return
        if v not in nodes: raise ValueError("Missing producer for "+v)
        idx=nodes[v]
        if idx in required: return
        required.add(idx)
        for inp in g.node[idx].input:
            if inp: walk(inp)
    for output_name in (source,scale_name,output): walk(output_name)
    ordered=[copy.deepcopy(n) for i,n in enumerate(g.node) if i in required]
    required_const=sorted({name for n in ordered for name in n.input if name in initial})
    shapes={}
    for name,vi in inputs.items():
        dims=[d.dim_value if d.HasField("dim_value") else (1 if i==0 else None)
              for i,d in enumerate(vi.type.tensor_type.shape.dim)]
        if len(dims)!=4 or any(not isinstance(x,int) or x<1 for x in dims):
            raise ValueError("Unknown model input dims {}: {}".format(name,dims))
        if vi.type.tensor_type.elem_type!=TensorProto.FLOAT:
            raise ValueError("Non-float ONNX input: "+name)
        shapes[name]=dims
    graph=helper.make_graph(ordered,"katago_rconv5_gpool_reference",
        [helper.make_tensor_value_info(name,TensorProto.FLOAT,dims) for name,dims in shapes.items()],
        [helper.make_tensor_value_info(source,TensorProto.FLOAT,[1,32,19,19]),
         helper.make_tensor_value_info(scale_name,TensorProto.FLOAT,[1,1,1,1]),
         helper.make_tensor_value_info(output,TensorProto.FLOAT,[1,96,1,1])],
        initializer=[copy.deepcopy(initial[x]) for x in required_const])
    ref=helper.make_model(graph,opset_imports=[helper.make_opsetid("",19)])
    ref.ir_version=9
    onnx.checker.check_model(ref)
    rng=np.random.default_rng(seed)
    feeds={name:(np.ones(dims,dtype=np.float32) if name=="InputMask" else
                 rng.normal(0,0.2,dims).astype(np.float32)) for name,dims in shapes.items()}
    opts=ort.SessionOptions()
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession(ref.SerializeToString(),opts,providers=["CPUExecutionProvider"])
    activation,scale,reference=sess.run([source,scale_name,output],feeds)
    if activation.shape!=(1,32,19,19) or reference.shape!=(1,96,1,1) or scale.size!=1:
        raise ValueError("Unexpected actual rconv5 global pool tensor shapes")
    if not np.isfinite(activation).all() or not np.isfinite(reference).all() or not np.isfinite(scale).all():
        raise ValueError("Non-finite reference data")
    s=float(scale.reshape(-1)[0])
    # Verify mathematically that masking and the 3-way aggregation can be
    # replaced by mean/max/scale when the board mask contains only ones.
    avg=activation.mean(axis=(2,3),keepdims=True)
    maxi=activation.max(axis=(2,3),keepdims=True)
    from_aggregates=np.concatenate([avg,avg*s,maxi],axis=1)
    # The reference may differ from numpy float32 due to the ONNX reduce order.
    gpool_output="rconv5/gpconcat/78"
    if gpool_output not in nodes: raise ValueError("Missing original global pool concat")
    # Reuse model with this additional output, preserving same initial random seed.
    graph.output.extend([helper.make_tensor_value_info(gpool_output,TensorProto.FLOAT,[1,96,1,1])])
    onnx.checker.check_model(ref)
    sess2=ort.InferenceSession(ref.SerializeToString(),opts,providers=["CPUExecutionProvider"])
    original_concat=sess2.run([gpool_output],feeds)[0]
    error=float(np.max(np.abs(from_aggregates-original_concat)))
    if error>0.002:
        raise ValueError("Reduced mean/max concat does not match ONNX gpool: maxerr="+str(error))

    path=Path(out_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<8s4I",b"K10GP001",19,19,32,96))
        f.write(np.asarray([s],dtype="<f2").tobytes())
        f.write(np.ascontiguousarray(activation.transpose(0,2,3,1),dtype="<f2").tobytes())
        f.write(np.ascontiguousarray(weights.transpose(0,2,3,1),dtype="<f2").tobytes())
        f.write(np.ascontiguousarray(reference.transpose(0,2,3,1),dtype="<f4").tobytes())
    expected=24+2*(1+19*19*32+96*96)+4*96
    if path.stat().st_size!=expected:
        raise ValueError("Incorrect export packet size {} vs {}".format(path.stat().st_size,expected))
    print("KATAGO_GPOOL_EXPORT node=rconv5/w1r original_ops={} gpool_scale={:.7f} concat_check_max_abs={:.7f}".format(len(ordered),s,error))
    print("KATAGO_GPOOL_EXPORT input={} w1r={} output={} bytes={}".format(activation.shape,weights.shape,reference.shape,expected))
    print("NOTE: real KataGo gpool branch only, not full rconv5 or full network.")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    export(a.model,a.out,a.seed)
