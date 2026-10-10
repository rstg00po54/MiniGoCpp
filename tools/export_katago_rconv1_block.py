#!/usr/bin/env python3
"""Export the full actual KataGo 10b rconv1 residual block as FP16 MDLA inputs.

Build an isolated ONNX CPU reference subgraph using the exact original ONNX
operators, 10b weights and normalisation constants. The source 10b model is
unchanged; the isolated reference uses opset 19/IR 9 for Python 3.8 support.

The only simplification for the Neuron graph is removing two multiplications
by InputMask, because this benchmark explicitly uses a full 19x19 mask of ones.

Binary format, all little-endian:
  8s magic K10RB001 + 4 uint32 (height,width,channels,kernel)
  FP16 NHWC block input [1,H,W,C]
  FP16 per-channel norm1.scale, norm1.bias [C] each
  FP16 OHWI conv1 weights [C,K,K,C]
  FP16 per-channel norm2.scale, norm2.bias [C] each
  FP16 OHWI conv2 weights [C,K,K,C]
  FP32 NHWC CPU golden residual output [1,H,W,C]
"""
import argparse
import copy
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def export(model_path, packet_path, seed):
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    graph = model.graph
    producer = {name: i for i, node in enumerate(graph.node) for name in node.output if name}
    initializers = {item.name: item for item in graph.initializer}
    initial_arrays = {name: numpy_helper.to_array(value) for name, value in initializers.items()}
    inputs = {value.name: value for value in graph.input}

    if "trunk/initbias/16" not in producer or "rconv1/27" not in producer:
        raise ValueError("Expected KataGo 10b trunk and rconv1 outputs are missing")
    input_name = "trunk/initbias/16"
    output_name = "rconv1/27"
    needed_nodes = set()
    def visit(value):
        if value in initializers or value in inputs:
            return
        if value not in producer:
            raise ValueError("Unresolved upstream value: " + value)
        index = producer[value]
        if index in needed_nodes:
            return
        needed_nodes.add(index)
        for dependency in graph.node[index].input:
            if dependency:
                visit(dependency)

    visit(input_name)
    visit(output_name)
    last = max(needed_nodes)
    nodes = [copy.deepcopy(graph.node[i]) for i in range(last+1) if i in needed_nodes]
    if not any(n.name == "rconv1" and output_name in n.output for n in nodes):
        raise ValueError("Expected rconv1 residual ADD was not found")
    used_constants = sorted({arg for node in nodes for arg in node.input if arg in initializers})
    input_shapes = {}
    for name, value in inputs.items():
        shape = [dim.dim_value if dim.HasField("dim_value") else
                 (1 if i == 0 else None) for i, dim in enumerate(value.type.tensor_type.shape.dim)]
        if len(shape) != 4 or any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError("Unsupported graph input shape {}: {}".format(name, shape))
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("Expected FP32 ONNX input: " + name)
        input_shapes[name] = shape

    weights = []
    norms = []
    for suffix in ("w1", "w2"):
        node = next((x for x in nodes if x.name == "rconv1/"+suffix), None)
        if node is None or node.op_type != "Conv":
            raise ValueError("Expected rconv1/{} convolution".format(suffix))
        attr = {x.name: helper.get_attribute_value(x) for x in node.attribute}
        if (list(attr.get("pads", [0,0,0,0])) != [1,1,1,1] or
            list(attr.get("strides", [1,1])) != [1,1] or
            list(attr.get("dilations", [1,1])) != [1,1] or
            int(attr.get("group", 1)) != 1):
            raise ValueError("Unexpected original convolution attributes: " + str(attr))
        weights.append(np.asarray(initial_arrays[node.input[1]], dtype=np.float32))
    for step in ("norm1", "norm2"):
        scale = np.asarray(initial_arrays["rconv1/"+step+".scale"], dtype=np.float32)
        bias = np.asarray(initial_arrays["rconv1/"+step+".bias"], dtype=np.float32)
        if scale.shape != (1,128,1,1) or bias.shape != (1,128,1,1):
            raise ValueError("Invalid norm {} scale/bias shapes: {},{}".format(step,scale.shape,bias.shape))
        norms.append((scale.reshape(-1), bias.reshape(-1)))

    h,w,c,k = 19,19,128,3
    if any(x.shape != (c,c,k,k) for x in weights):
        raise ValueError("Unexpected KataGo residual weights: " + str([x.shape for x in weights]))
    if input_shapes.get("InputMask") != [1,1,h,w]:
        raise ValueError("Residual mask shape is not 19x19")
    graph_outputs = [
        helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1,c,h,w]),
        helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1,c,h,w]),
    ]
    reference = helper.make_graph(
        nodes,"katago_rconv1_fp32_reference",
        [helper.make_tensor_value_info(name,TensorProto.FLOAT,shape)
         for name,shape in input_shapes.items()],
        graph_outputs,initializer=[copy.deepcopy(initializers[name]) for name in used_constants])
    reference_model=helper.make_model(reference,opset_imports=[helper.make_opsetid("",19)])
    reference_model.ir_version=9
    onnx.checker.check_model(reference_model)
    rng=np.random.default_rng(seed)
    feed={name:(np.ones(shape,dtype=np.float32) if name=="InputMask" else
                rng.normal(0,0.2,size=shape).astype(np.float32))
          for name,shape in input_shapes.items()}
    if not np.all(feed["InputMask"] == 1):
        raise ValueError("InputMask must be all ones to remove mask multiplications")
    options=ort.SessionOptions()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(reference_model.SerializeToString(),options,providers=["CPUExecutionProvider"])
    raw_input,raw_output=session.run([input_name,output_name],feed)
    if raw_input.shape!=(1,c,h,w) or raw_output.shape!=(1,c,h,w):
        raise ValueError("Unexpected reference block shapes: {} {}".format(raw_input.shape,raw_output.shape))
    src=np.ascontiguousarray(raw_input.transpose(0,2,3,1).astype("<f2"))
    golden=np.ascontiguousarray(raw_output.transpose(0,2,3,1).astype("<f4"))

    path=Path(packet_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8s4I",b"K10RB001",h,w,c,k))
        stream.write(src.tobytes())
        for i in range(2):
            stream.write(np.ascontiguousarray(norms[i][0].astype("<f2")).tobytes())
            stream.write(np.ascontiguousarray(norms[i][1].astype("<f2")).tobytes())
            stream.write(np.ascontiguousarray(weights[i].transpose(0,2,3,1).astype("<f2")).tobytes())
        stream.write(golden.tobytes())
    expected=24+2*(h*w*c+4*c+2*c*k*k*c)+4*h*w*c
    if path.stat().st_size!=expected:
        raise ValueError("Invalid packet length: {} vs {}".format(path.stat().st_size,expected))
    print("KATAGO_BLOCK_EXPORT block=rconv1 graph_nodes={} convs=2 residual_add=1".format(len(nodes)))
    print("KATAGO_BLOCK_EXPORT input={} output={} weights={} mask=all_ones".format(raw_input.shape,raw_output.shape,weights[0].shape))
    print("KATAGO_BLOCK_EXPORT packet={} bytes={} CPU_FP32_min={:.6f} max={:.6f}".format(
        path,path.stat().st_size,float(golden.min()),float(golden.max())))
    print("NOTE: original graph preserved in CPU reference; MDLA graph omits multiplication by all-ones masks.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    export(args.model,args.out,args.seed)
