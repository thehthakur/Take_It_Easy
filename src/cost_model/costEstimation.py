import onnx
from onnx2torch import convert
from calflops import calculate_flops
from onnx import ModelProto
import argparse
import os
import sys
import torch
import numpy as np

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
sys.path.insert(0, rootPath)

# onnx_path = "assets/onnx_files/example_1_initial_model.onnx"
# onnx_path = "assets/onnx_files/example_1_transformation_1_enlarge_conv_kernel.onnx"
# onnx_model = onnx.load(onnx_path)
# print(f"Loaded ONNX model from {onnx_path}")

# torch_model = convert(onnx_model)
# print("Converted ONNX model to PyTorch")

# batch_size = 1
# input_shape = (batch_size, 256, 64, 64)

# flops, macs, params = calculate_flops(
#     model=torch_model,
#     input_shape=input_shape,
#     output_as_string=True,
#     output_precision=4
# )

# print("Model FLOPs: %s   MACs: %s   Params: %s" % (flops, macs, params))

def get_flops(model: ModelProto):
    torch_model = convert(model)

    input_shapes = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)
        input_shapes[name] = tuple(shape)

    if not input_shapes:
        raise ValueError("No input shapes found in the ONNX model.")

    print(f"Inferred input shapes: {input_shapes}")
    inputs = []
    for value in input_shapes.values():
        inputs.append(value)

    # input_tensors = {name: torch.rand(*shape) for name, shape in input_shapes.items()}

    flops, macs, params = calculate_flops(
        model=torch_model,
        input_shape=tuple(inputs[0]),
        output_as_string=True,
        output_precision=4
    )
    
    # Print the results
    print("Model FLOPs: %s   MACs: %s   Params: %s" % (flops, macs, params))
    
    # Return FLOPs for further use
    return flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an ONNX model to Netron.")
    parser.add_argument("path", type=str, help="Path to the ONNX model file from root folder")
    args = parser.parse_args()

    model_path = os.path.join(rootPath, args.path)
    model = onnx.load_model(model_path)
 
    flops = get_flops(model)
    print(flops)
    