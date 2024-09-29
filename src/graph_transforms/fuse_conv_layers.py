import numpy as np
import onnx
from onnx import numpy_helper
from typing import List

def fuse_conv_layers(model: onnx.onnx_ml_pb2.ModelProto, conv1_identifier: str, conv2_identifier: str) -> onnx.onnx_ml_pb2.ModelProto:
    graph = model.graph
    print(graph)
    return model

# example usage
if __name__ == '__main__':
    model = onnx.load("../../assets/onnx_files/example_1_transformation_1_enlarge_conv_kernel.onnx")
    transformed_model = fuse_conv_layers(model=model, conv1_identifier="conv1", conv2_identifier="conv2")
