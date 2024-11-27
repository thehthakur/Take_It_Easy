import os
import sys
import argparse
import onnx
from onnx import ModelProto
from typing import List, Any
import heapq
from cost_model.costModel import returnFLOPs, returnMem
from cost_model.compute_cost import compute_cost

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
sys.path.insert(0, rootPath)

def extract_node_attributes(node, model):
    shape_info = onnx.shape_inference.infer_shapes(model)

    if node.op_type == "Add":
        input_name = node.input

        for _, ndd in enumerate(shape_info.graph.value_info):
            for input_n in input_name:
                if ndd.name == input_n:
                    dims = tuple(dim.dim_value for dim in ndd.type.tensor_type.shape.dim)
                    return dims
    
        return None
    
    elif node.op_type == "Relu":
        input_name = node.input

        for _, ndd in enumerate(shape_info.graph.value_info):
            for input_n in input_name:
                if ndd.name == input_n:
                    dims = tuple(dim.dim_value for dim in ndd.type.tensor_type.shape.dim)
                    return dims
        return None
    
    elif node.op_type == "Conv":
        input_name = node.input
        output_name = node.output
        kernel_size = next(dim.ints[0] for dim in node.attribute if dim.name == "kernel_shape")
        pads = next(dim.ints[0] for dim in node.attribute if dim.name == "pads")
        strides = next(dim.ints[0] for dim in node.attribute if dim.name == "strides")
        # print(model.graph.input)
        # print(node)
        dims = ()

        for ndd in model.graph.input:
            for nddd in input_name:
                if nddd == ndd.name:
                    dimss = tuple(dim.dim_value for dim in ndd.type.tensor_type.shape.dim)
                    dims = dims + (dimss[2], dimss[1])

        num_filters = None
        for _, ndd in enumerate(shape_info.graph.value_info):
            # print(ndd)
            for nddd in input_name:
                if nddd == ndd.name:
                    dimss = tuple(dim.dim_value for dim in ndd.type.tensor_type.shape.dim)
                    dims = dims + (dimss[2], dimss[1])
            for nddd in output_name:
                if nddd == ndd.name:
                    dimss = tuple(dim.dim_value for dim in ndd.type.tensor_type.shape.dim)
                    num_filters = dimss[1]
        
        if len(dims) != 0:
            dims = dims + (kernel_size, num_filters, pads, strides)
            
        # print(dims)
        return dims


def adjacency_graph(model: ModelProto):
    adjacency_list = {}

    for input_tensor in model.graph.input:
        input_name = input_tensor.name
        if input_name not in adjacency_list:
            adjacency_list[input_name] = []

        for node in model.graph.node:
            if input_name in node.input:
                node_name = node.name if node.name else node.output[0]
                adjacency_list[input_name].append(node_name)

    for node in model.graph.node:
        node_name = node.name if node.name else node.output[0]
        if node_name not in adjacency_list:
            adjacency_list[node_name] = []

        for output in node.output:
            for other_node in model.graph.node:
                if output in other_node.input:
                    connected_node_name = other_node.name if other_node.name else other_node.output[0]
                    adjacency_list[node_name].append(connected_node_name)

    for output_tensor in model.graph.output:
        output_name = output_tensor.name
        if output_name not in adjacency_list:
            adjacency_list[output_name] = []

        for node in model.graph.node:
            if output_name in node.output:
                node_name = node.name if node.name else node.output[0]
                adjacency_list[node_name].append(output_name)

    return adjacency_list

def reverse_adjacency_graph(adj_list):
    reversed_adj_list = {node: [] for node in adj_list}
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            reversed_adj_list[neighbor].append(node)
    return reversed_adj_list

def calculate_indegree(adj_list):
    indegree = {node: 0 for node in adj_list}
    for neighbors in adj_list.values():
        for neighbor in neighbors:
            indegree[neighbor] += 1
    return indegree

def get_name_to_node(model: ModelProto) -> dict[str, str]:
    name_to_node = {}
    for node in model.graph.node:
        name_to_node[node.name] = node.op_type
    return name_to_node

def get_node_to_flops(model: ModelProto) -> dict[str, int]:
    node_to_flops = {}
    for node in model.graph.input:
        node_to_flops[node.name] = 0
    for node in model.graph.output:
        node_to_flops[node.name] = 0
    for node in model.graph.node:
        node_attrs = extract_node_attributes(node, model)
        node_to_flops[node.name] = returnFLOPs(node.op_type, node_attrs)
    return node_to_flops

def get_node_to_mem(model: ModelProto) -> dict[str, int]:
    node_to_flops = {}
    for node in model.graph.input:
        node_to_flops[node.name] = 0
    for node in model.graph.output:
        node_to_flops[node.name] = 0
    for node in model.graph.node:
        node_attrs = extract_node_attributes(node, model)
        node_to_flops[node.name] = returnMem(node.op_type, node_attrs)
    return node_to_flops

def calculate_cost(model: ModelProto):
    name_to_node = get_name_to_node(model)
    node_to_cost = get_node_to_flops(model)
    node_to_mem_cost = get_node_to_mem(model)
    adj_list = adjacency_graph(model)
    parent = reverse_adjacency_graph(adj_list)
    indegree = calculate_indegree(adj_list)

    # print(name_to_node)
    # print(node_to_cost)
    # print(adj_list)
    # print(parent)
    # print(indegree)

    cost = compute_cost(len(name_to_node), adj_list, parent, indegree, node_to_cost, node_to_mem_cost)
    return cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an ONNX model to Netron.")
    parser.add_argument("path", type=str, help="Path to the ONNX model file from root folder")
    args = parser.parse_args()

    model_path = os.path.join(rootPath, args.path)
    model = onnx.load_model(model_path)

    # print(adjacency_graph(model))
    print(calculate_cost(model))