import os
import sys
import argparse
import onnx
from onnx import ModelProto
from graph_transforms.enlarge_conv_kernel import enlarge_conv_kernel
from inference import inference_time
from typing import List, Any
import heapq

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

def apply_operation(model: ModelProto, operation):
    graph = model.graph

    new_models = []

    for node in graph.node:
        if node.op_type == "Conv":
            new_model = operation(model, node.name, [3,3])
            new_models.append(new_model)

    return new_models

def backtrack(model: ModelProto, cost_model, alpha: float, operations: List[Any]):
    queue = []
    heapq.heappush(queue, (cost_model(model), model))
    optimized_model = model

    while len(queue):
        _, curr_model = heapq.heappop(queue)

        for operation in operations:
            new_models = apply_operation(curr_model, operation)

            for mdl in new_models:
                mdl_cost = cost_model(mdl)
                if mdl_cost < cost_model(optimized_model):
                    optimized_model = mdl
                elif mdl_cost < alpha * cost_model(optimized_model):
                    heapq.heappush(queue, (mdl_cost, mdl))

    return optimized_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an ONNX model to Netron.")
    parser.add_argument("path", type=str, help="Path to the ONNX model file from root folder")
    args = parser.parse_args()

    model_path = os.path.join(rootPath, args.path)
    model = onnx.load_model(model_path)

    alpha = 20
    optimized_model = backtrack(model, inference_time, alpha, [enlarge_conv_kernel])
    onnx.save(optimized_model, "optimized_model.onnx")