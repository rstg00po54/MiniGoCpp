#!/usr/bin/env python3
"""Print first nodes of KataGo's real ONNX topology; helps plan the Neuron MDLA residual block."""
import argparse
import onnx
from onnx import numpy_helper, helper

p=argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--limit",type=int,default=100)
args=p.parse_args()
g=onnx.load(args.model).graph
initializers={x.name:tuple(x.dims) for x in g.initializer}
print("KATAGO_GRAPH_INSPECT inputs:",[(x.name,[(d.dim_value if d.HasField('dim_value') else d.dim_param) for d in x.type.tensor_type.shape.dim]) for x in g.input])
print("KATAGO_GRAPH_INSPECT total_nodes:",len(g.node),"initializers:",len(g.initializer))
for i,n in enumerate(g.node):
    if i>=args.limit: break
    attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
    weights=[(inp,initializers[inp]) for inp in n.input if inp in initializers]
    print("KATAGO_NODE {:03d} {:<18} {:<32} IN={} OUT={} WEIGHTS={} ATTR={}".format(
        i,n.op_type,n.name,";".join(n.input[:5]),";".join(n.output[:3]),weights,attrs))
