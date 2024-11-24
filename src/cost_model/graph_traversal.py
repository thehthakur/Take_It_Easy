import os
import sys
import argparse
import onnx
from onnx import ModelProto
from typing import List, Any
import heapq

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
sys.path.insert(0, rootPath)

def adjacency_graph(modeL: ModelProto):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an ONNX model to Netron.")
    parser.add_argument("path", type=str, help="Path to the ONNX model file from root folder")
    args = parser.parse_args()

    model_path = os.path.join(rootPath, args.path)
    model = onnx.load_model(model_path)

    print(adjacency_graph(model))