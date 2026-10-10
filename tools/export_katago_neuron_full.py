#!/usr/bin/env python3
"""Convert actual KataGo 10b 19x19 ONNX to ONE full-network native FP16 Neuron test packet.

For an all-valid 19x19 board (InputMask=ones), constant-fold mask-only ops,
replace spatial ReduceSum / mask-cell-count Div with Mean, convert all ONNX
NCHW tensors to NHWC and Conv filters from OIHW to OHWI, and preserve all five
original network output heads. Does NOT modify original model on disk.

Use the original 198-node graph as a standalone opset19 / IR9 ONNX Runtime CPU
reference (required by Python 3.8). The emitted packet includes the whole
converted graph, original FP16 weights, real FP16 feature inputs, and FP32
reference outputs. A pinned Neuron device must support/compile all ops.
"""
import argparse
import copy
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

# NNAPI / MediaTek Neuron operation IDs, using the same ABI as previous tests.
ADD, CONCAT, CONV, MUL, RELU, MEAN, REDUCE_MAX = 0, 2, 3, 18, 19, 31, 77
TYPE_INT32, TYPE_TENSOR_INT32, TYPE_BOOL, TYPE_FP16 = 1, 4, 6, 8
FLAG_CONST, FLAG_INPUT, FLAG_OUTPUT = 1, 2, 4


def run(model_path, out_path, seed=42):
    model=onnx.load(str(model_path),load_external_data=True)
    onnx.checker.check_model(model)
    for value in model.graph.input:
        shape=value.type.tensor_type.shape.dim
        if not shape or (shape[0].dim_param and shape[0].dim_param != "batch"):
            raise ValueError("Unexpected dynamic ONNX batch input "+value.name)
        shape[0].ClearField("dim_param")
        shape[0].dim_value=1
    reference=copy.deepcopy(model)
    reference.ir_version=9
    del reference.opset_import[:]
    reference.opset_import.append(helper.make_opsetid("",19))
    onnx.checker.check_model(reference)
    inferred=shape_inference.infer_shapes(reference)
    dims={}
    for value in list(inferred.graph.input)+list(inferred.graph.value_info)+list(inferred.graph.output):
        t=value.type.tensor_type
        if t.elem_type != TensorProto.FLOAT:
            continue
        shape=[int(x.dim_value) if x.HasField("dim_value") else None for x in t.shape.dim]
        if all(x is not None and x>0 for x in shape):
            dims[value.name]=tuple(shape)
    initial={x.name:numpy_helper.to_array(x) for x in reference.graph.initializer}
    input_shapes={}
    for vi in reference.graph.input:
        if vi.name not in dims:raise ValueError("Missing static model input dimensions for "+vi.name)
        input_shapes[vi.name]=dims[vi.name]
    if input_shapes.get("InputSpatial")!=(1,22,19,19) or input_shapes.get("InputGlobal")!=(1,19,1,1):
        raise ValueError("Unexpected actual KataGo 10b model inputs: "+str(input_shapes))
    if input_shapes.get("InputMask")!=(1,1,19,19):
        raise ValueError("Expected 19x19 full-board mask")

    rng=np.random.default_rng(seed)
    feeds={name:(np.ones(shape,np.float32) if name=="InputMask" else
                 rng.normal(0,0.2,size=shape).astype(np.float32))
           for name,shape in input_shapes.items()}
    opts=ort.SessionOptions()
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(reference.SerializeToString(),opts,providers=["CPUExecutionProvider"])
    output_names=[x.name for x in session.get_outputs()]
    golden=session.run(output_names,feeds)
    if len(golden)!=5:
        raise ValueError("Expected all 5 KataGo 10b outputs, found "+str(output_names))
    print("KATAGO_FULL_CPU_REF outputs",[(n,tuple(x.shape),float(x.min()),float(x.max()))
                                          for n,x in zip(output_names,golden)])

    # Constant propagation for all-ones mask scalars and zero mask shifts.
    constants={key:np.asarray(value) for key,value in initial.items()}
    constants["InputMask"]=feeds["InputMask"]
    skipped={}  # aliases (Identity, multiplication by ones, addition of zero)
    producer={out:node for node in reference.graph.node for out in node.output if out}

    def resolve(name):
        while name in skipped and skipped[name]!=name:
            name=skipped[name]
        return name

    def constant_eval(node, args):
        attrs={x.name:helper.get_attribute_value(x) for x in node.attribute}
        if node.op_type=="Add":return args[0]+args[1]
        if node.op_type=="Mul":return args[0]*args[1]
        if node.op_type=="Div":return args[0]/args[1]
        if node.op_type=="Sqrt":return np.sqrt(args[0])
        if node.op_type=="Relu":return np.maximum(args[0],0)
        if node.op_type=="Identity":return args[0]
        if node.op_type=="ReduceSum":
            axes=tuple(int(x) for x in np.asarray(args[1]).flatten())
            return np.sum(args[0],axis=axes,keepdims=bool(attrs.get("keepdims",1)))
        if node.op_type=="ReduceMean":
            axes=tuple(int(x) for x in np.asarray(args[1]).flatten())
            return np.mean(args[0],axis=axes,keepdims=bool(attrs.get("keepdims",1)))
        if node.op_type=="ReduceMax":
            axes=tuple(int(x) for x in np.asarray(args[1]).flatten())
            return np.max(args[0],axis=axes,keepdims=bool(attrs.get("keepdims",1)))
        if node.op_type=="Concat":
            return np.concatenate(args,axis=attrs["axis"])
        raise ValueError("Unable to constant-fold "+node.op_type)

    for node in reference.graph.node:
        args=[constants.get(resolve(x)) for x in node.input]
        if len(node.output)!=1:raise ValueError("Multi-output op unsupported: "+node.name)
        if all(x is not None for x in args):
            try:constants[node.output[0]]=np.asarray(constant_eval(node,args))
            except (ValueError,TypeError,IndexError):
                pass
        elif node.op_type=="Identity":
            skipped[node.output[0]]=resolve(node.input[0])
        elif node.op_type in ("Mul","Add"):
            for j,x in enumerate(args):
                if x is None:continue
                if (node.op_type=="Mul" and np.all(x==1)) or (node.op_type=="Add" and np.all(x==0)):
                    skipped[node.output[0]]=resolve(node.input[1-j])

    # Use an explicitly typed, self-contained packet; no ad-hoc Android
    # ONNX graph parser, no SDK headers from the phone.
    operands=[];operand_by_name={};operations=[];input_ids=[];output_ids=[]
    def add_operand(name,typ,shape,raw=b"",flags=0):
        if name in operand_by_name:
            return operand_by_name[name]
        if len(shape)>4:
            raise ValueError("Operand dimensions exceed rank4: {} {}".format(name,shape))
        shape=tuple(int(i) for i in shape)
        if any(x<=0 for x in shape):raise ValueError("Invalid operand "+name+str(shape))
        idx=len(operands)
        operands.append((name,typ,shape,flags,bytes(raw)))
        operand_by_name[name]=idx
        return idx

    def nhwc_shape(shape):
        if len(shape)==4:return (shape[0],shape[2],shape[3],shape[1])
        return shape

    def tensor_bytes(array,conv_filter=False):
        a=np.asarray(array,dtype=np.float32)
        if a.ndim==4:
            if conv_filter:a=a.transpose(0,2,3,1)
            else:a=a.transpose(0,2,3,1)
        return np.ascontiguousarray(a,dtype="<f2").tobytes()

    def constant_operand(name,conv_filter=False):
        name=resolve(name)
        if name in operand_by_name:return operand_by_name[name]
        a=constants[name]
        if a.dtype.kind in "iu":
            raise ValueError("Unexpected integer constant operand in converted data flow: "+name)
        shape=nhwc_shape(a.shape)
        return add_operand(name,TYPE_FP16,shape,tensor_bytes(a,conv_filter),FLAG_CONST)

    def variable(name):
        name=resolve(name)
        if name in operand_by_name:return operand_by_name[name]
        if name in constants:return constant_operand(name)
        if name not in dims:
            raise ValueError("Missing inferred ONNX tensor shape for "+name)
        return add_operand(name,TYPE_FP16,nhwc_shape(dims[name]))

    def const_int(value,tag="scalar"):
        name="@{}:{}".format(tag,value)
        return add_operand(name,TYPE_INT32,(),struct.pack("<i",int(value)),FLAG_CONST)
    def const_int_vec(values):
        arr=tuple(int(x) for x in values)
        name="@axes:"+",".join(map(str,arr))
        raw=np.asarray(arr,dtype="<i4").tobytes()
        return add_operand(name,TYPE_TENSOR_INT32,(len(arr),),raw,FLAG_CONST)
    def const_bool(value):
        return add_operand("@bool:"+str(value),TYPE_BOOL,(),bytes([int(bool(value))]),FLAG_CONST)
    def zero_bias(channels):
        return add_operand("@zero_bias:"+str(channels),TYPE_FP16,(channels,),
                           np.zeros(channels,dtype="<f2").tobytes(),FLAG_CONST)
    def add_operation(op,input_ids_,output_name):
        out=variable(output_name)
        operations.append((op,list(input_ids_),[out]))
        return out

    for name in ("InputSpatial","InputGlobal"):
        a=feeds[name]
        idx=add_operand(name,TYPE_FP16,nhwc_shape(a.shape),tensor_bytes(a),FLAG_INPUT)
        input_ids.append(idx)

    converted=[]
    for node in reference.graph.node:
        target=node.output[0]
        if target in constants or target in skipped:continue
        typ=node.op_type
        source=[resolve(n) for n in node.input]
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        if typ=="Conv":
            x=variable(source[0])
            weights=np.asarray(constants[source[1]])
            if len(weights.shape)!=4:raise ValueError("Conv weights rank !=4 "+node.name)
            out_ch,in_ch,kh,kw=weights.shape
            weight=constant_operand(source[1],conv_filter=True)
            bias=constant_operand(source[2]) if len(source)>=3 and source[2] else zero_bias(out_ch)
            stride=attrs.get("strides",[1,1]);dilation=attrs.get("dilations",[1,1]);pads=attrs.get("pads",[0,0,0,0])
            if (list(stride)!=[1,1] or list(dilation)!=[1,1] or
                int(attrs.get("group",1))!=1 or len(pads)!=4):
                raise ValueError("Unsupported Conv form in "+node.name+":"+str(attrs))
            pt,pl,pb,pr=map(int,pads)
            ins=[x,weight,bias,const_int(pl),const_int(pr),const_int(pt),const_int(pb),
                 const_int(1),const_int(1),const_int(0)]
            add_operation(CONV,ins,target)
        elif typ in ("Add","Mul"):
            ins=[variable(source[0]),variable(source[1]),const_int(0)]
            add_operation(ADD if typ=="Add" else MUL,ins,target)
        elif typ=="Relu":
            add_operation(RELU,[variable(source[0])],target)
        elif typ in ("ReduceMax","ReduceMean"):
            axes=np.asarray(constants[source[1]]).flatten().tolist()
            if axes!=[2,3]:
                raise ValueError("Unexpected reduction axes in "+node.name+":"+str(axes))
            if int(attrs.get("keepdims",1))!=1:raise ValueError("Reduce must keep dimensions")
            # NHWC spatial axes.
            ins=[variable(source[0]),const_int_vec([1,2]),const_bool(True) if typ=="ReduceMax" else const_int(1)]
            add_operation(REDUCE_MAX if typ=="ReduceMax" else MEAN,ins,target)
        elif typ=="Div":
            # The 19x19 full mask makes upstream spatial SUM / mask pixel count
            # exactly a spatial MEAN, as validated in the real rconv5 GPool test.
            src_node=producer.get(source[0])
            denom=constants.get(source[1])
            if src_node is None or src_node.op_type!="ReduceSum" or denom is None or np.size(denom)!=1:
                raise ValueError("Unsupported non-GPool Div "+node.name)
            if not np.allclose(denom,361.0):raise ValueError("Expected 361 valid board points "+node.name)
            axes=np.asarray(constants[resolve(src_node.input[1])]).flatten().tolist()
            if axes!=[2,3]:
                raise ValueError("Unexpected spatial reduction axes in Div "+node.name)
            ins=[variable(src_node.input[0]),const_int_vec([1,2]),const_int(1)]
            add_operation(MEAN,ins,target)
        elif typ=="ReduceSum":
            # Materialized only by the rewritten downstream Div, no other consumer.
            consumers=[n for n in reference.graph.node if target in n.input]
            if len(consumers)!=1 or consumers[0].op_type!="Div":
                raise ValueError("Unmatched ReduceSum: "+node.name)
            converted.append(("defer_mean",node.name))
            continue
        elif typ=="Concat":
            if attrs.get("axis",None)!=1:raise ValueError("Only NCHW channel Concat supported "+node.name)
            ins=[variable(x) for x in source]+[const_int(3)]
            add_operation(CONCAT,ins,target)
        else:
            raise ValueError("Unconverted KataGo operator {} ({})".format(node.name,typ))
        converted.append((typ,node.name))
    # Model graph outputs are Identity aliases of actual Neuron outputs.
    for name in output_names:
        idx=variable(name)
        operands[idx]=(operands[idx][0],operands[idx][1],operands[idx][2],
                       operands[idx][3]|FLAG_OUTPUT,operands[idx][4])
        output_ids.append(idx)
    if len(output_ids)!=5:raise ValueError("Expected five outputs")
    if len(operations)<100:raise ValueError("Unreasonably few full-network Neuron ops "+str(len(operations)))

    path=Path(out_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<8s4I",b"K10NW001",len(operands),len(operations),len(input_ids),len(output_ids)))
        for name,typ,shape,flags,raw in operands:
            padded=tuple(shape)+(0,)*(4-len(shape))
            out.write(struct.pack("<8I",typ,len(shape),flags,len(raw),*padded))
            out.write(raw)
        for operation,ins,outs in operations:
            out.write(struct.pack("<3I",operation,len(ins),len(outs)))
            out.write(struct.pack("<{}I".format(len(ins)),*ins))
            out.write(struct.pack("<{}I".format(len(outs)),*outs))
        for idx in input_ids:out.write(struct.pack("<I",idx))
        for idx in output_ids:out.write(struct.pack("<I",idx))
        for name,g in zip(output_names,golden):
            name_bytes=name.encode("utf8")
            out.write(struct.pack("<I",len(name_bytes)))
            out.write(name_bytes)
            data=np.ascontiguousarray(g.transpose(0,2,3,1) if g.ndim==4 else g,dtype="<f4").tobytes()
            out.write(struct.pack("<I",len(data)))
            out.write(data)
    print("KATAGO_FULL_NEURON_EXPORT onnx_nodes={} converted_ops={} operands={} inputs={} outputs={} bytes={}".format(
        len(reference.graph.node),len(operations),len(operands),len(input_ids),len(output_ids),path.stat().st_size))
    print("KATAGO_FULL_NEURON_OUTPUTS",[(n,tuple(x.shape)) for n,x in zip(output_names,golden)])
    print("Full original 10b network; test input uses all-ones 19x19 board mask. No hardware claims until tested.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    run(args.model,args.out,args.seed)
