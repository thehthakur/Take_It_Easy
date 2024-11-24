import os
import sys
import argparse
import onnx
from onnx import ModelProto
from typing import List, Any
import heapq

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

def adjacency_graph(modeL: ModelProto):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an ONNX model to Netron.")
    parser.add_argument("path", type=str, help="Path to the ONNX model file from root folder")
    args = parser.parse_args()

    model_path = os.path.join(rootPath, args.path)
    model = onnx.load_model(model_path)

    adjacency_graph(model)