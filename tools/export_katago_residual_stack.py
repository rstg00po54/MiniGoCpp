#!/usr/bin/env python3
"""Pack actual KataGo 10b rconv1..rconvN (N=1..4) for one FP16 Neuron graph.

CPU golden is computed by the original upstream ONNX subgraph in opset19 / IR9
to support Python 3.8. Board InputMask is all ones, hence Neuron omits identity
mask-multiplications. The original model, weights and residual topology remain unchanged.
Packet: magic K10RS001, five uint32 H W C kernel N, FP16 NHWC input,
per block: FP16 scale1,bias1,OHWI weights1,scale2,bias2,OHWI weights2,
finally FP32 NHWC CPU golden output. NOT a full KataGo evaluation.
"""
import argparse
import copy
import struct
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto


def export(model_path, out_path, count, seed=42):
    if not 1 <= count <= 4:
        raise ValueError("Only the four standard residual blocks before rconv5's global pooling are supported")
    model = onnx.load(model_path, load_external_data=True)
    onnx.checker.check_model(model)
    graph = model.graph
    producers = {value: i for i, node in enumerate(graph.node) for value in node.output if value}
    initial = {x.name: x for x in graph.initializer}
    arrays = {name: numpy_helper.to_array(x) for name,x in initial.items()}
    external = {v.name: v for v in graph.input}
    start = "trunk/initbias/16"
    end_node = next((n for n in graph.node if n.name == "rconv{}".format(count) and n.op_type == "Add"), None)
    if start not in producers or end_node is None:
        raise ValueError("The expected KataGo trunk/rconv{} topology was not found".format(count))
    end = end_node.output[0]

    seen = set()
    def visit(name):
        if name in initial or name in external:
            return
        if name not in producers:
            raise ValueError("Unresolved ONNX input: " + name)
        idx = producers[name]
        if idx in seen:
            return
        seen.add(idx)
        for arg in graph.node[idx].input:
            if arg: visit(arg)
    visit(start)
    visit(end)
    nodes = [copy.deepcopy(graph.node[i]) for i in range(len(graph.node)) if i in seen]
    names = {n.name for n in nodes}
    for b in range(1,count+1):
        needed = ("rconv{}/w1".format(b), "rconv{}/w2".format(b), "rconv{}".format(b))
        if not all(n in names for n in needed):
            raise ValueError("Residual block {} does not have expected two-convolution architecture".format(b))
    used_initial = sorted({arg for n in nodes for arg in n.input if arg in initial})

    shapes={}
    for name,val in external.items():
        if val.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("Only FLOAT ONNX inputs supported")
        dims=[dim.dim_value if dim.HasField("dim_value") else (1 if i==0 else None)
              for i,dim in enumerate(val.type.tensor_type.shape.dim)]
        if len(dims)!=4 or any(not isinstance(v,int) or v<1 for v in dims):
            raise ValueError("Unexpected ONNX input shape {}: {}".format(name,dims))
        shapes[name]=dims
    h=w=19;c=128;k=3
    if shapes.get("InputMask") != [1,1,h,w]:
        raise ValueError("Expected 19x19 InputMask")
    params=[]
    for b in range(1,count+1):
        stage=[]
        for norm,weight_name in (("norm1","w1"),("norm2","w2")):
            prefix="rconv{}/".format(b)
            scale=np.asarray(arrays[prefix+norm+".scale"],dtype=np.float32)
            bias=np.asarray(arrays[prefix+norm+".bias"],dtype=np.float32)
            wnode=next(n for n in nodes if n.name==prefix+weight_name)
            weight=np.asarray(arrays[wnode.input[1]],dtype=np.float32)
            attrs={a.name:helper.get_attribute_value(a) for a in wnode.attribute}
            if (scale.shape!=(1,c,1,1) or bias.shape!=(1,c,1,1) or
                weight.shape!=(c,c,k,k) or wnode.op_type!="Conv" or
                list(attrs.get("pads",[0,0,0,0]))!=[1,1,1,1] or
                list(attrs.get("strides",[1,1]))!=[1,1] or
                list(attrs.get("dilations",[1,1]))!=[1,1] or int(attrs.get("group",1))!=1):
                raise ValueError("Unexpected standard residual block configuration in "+prefix+weight_name)
            stage.extend([np.ascontiguousarray(scale.reshape(-1),dtype="<f2"),
                          np.ascontiguousarray(bias.reshape(-1),dtype="<f2"),
                          np.ascontiguousarray(weight.transpose(0,2,3,1),dtype="<f2")])
        params.append(stage)

    reference=helper.make_graph(
        nodes,"katago_first_{}_residual_blocks".format(count),
        [helper.make_tensor_value_info(name,TensorProto.FLOAT,shape) for name,shape in shapes.items()],
        [helper.make_tensor_value_info(start,TensorProto.FLOAT,[1,c,h,w]),
         helper.make_tensor_value_info(end,TensorProto.FLOAT,[1,c,h,w])],
        initializer=[copy.deepcopy(initial[name]) for name in used_initial])
    submodel=helper.make_model(reference,opset_imports=[helper.make_opsetid("",19)])
    submodel.ir_version=9
    onnx.checker.check_model(submodel)
    rng=np.random.default_rng(seed)
    feeds={name:(np.ones(shape,dtype=np.float32) if name=="InputMask" else
                 rng.normal(0,0.2,size=shape).astype(np.float32)) for name,shape in shapes.items()}
    session_opts=ort.SessionOptions()
    session_opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(submodel.SerializeToString(),sess_options=session_opts,providers=["CPUExecutionProvider"])
    first, golden=session.run([start,end],feeds)
    if first.shape!=(1,c,h,w) or golden.shape!=(1,c,h,w):
        raise ValueError("Unexpected subgraph IO: {}, {}".format(first.shape,golden.shape))
    packet=Path(out_path)
    packet.parent.mkdir(parents=True,exist_ok=True)
    with packet.open("wb") as output:
        output.write(struct.pack("<8s5I",b"K10RS001",h,w,c,k,count))
        output.write(np.ascontiguousarray(first.transpose(0,2,3,1),dtype="<f2").tobytes())
        for stage in params:
            for arr in stage: output.write(arr.tobytes())
        output.write(np.ascontiguousarray(golden.transpose(0,2,3,1),dtype="<f4").tobytes())
    expected=28+2*(h*w*c+count*(4*c+2*c*k*k*c))+4*h*w*c
    assert packet.stat().st_size==expected,(packet.stat().st_size,expected)
    print("KATAGO_STACK_EXPORT blocks={} ops={} convs={} cpu_graph_nodes={} packet_bytes={}".format(
          count,count*7,count*2,len(nodes),expected))
    print("KATAGO_STACK_EXPORT input={} output={} CPU_FP32_min={:.6f} max={:.6f}".format(
          first.shape,golden.shape,float(golden.min()),float(golden.max())))
    print("Mask all ones; the native graph omits two identity mask multiplications per block.")
    print("NOTE: this is the first {} residual blocks only, NOT KataGo 10b NN evals/s.".format(count))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    export(a.model,a.out,a.blocks,a.seed)
