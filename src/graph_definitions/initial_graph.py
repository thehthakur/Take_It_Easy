import numpy as np
import onnx
from onnx import helper, TensorProto

from nodes.conv import ConvLayer
from nodes.relu import ReluLayer
from nodes.add import AddLayer

# Define an input and output tensor with a specific NCHW
N_in = 1
C_in = 256
H_in = 64
W_in = 64
model_input_name = "input"
input_tensor = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, [N_in, C_in, H_in, W_in])

N_out = 1
C_out = 256
H_out = 64
W_out = 64
model_output_name = "Output"
output_tensor = onnx.helper.make_tensor_value_info(model_output_name, TensorProto.FLOAT, [N_out, C_out, H_out, W_out])

# node1
conv1 = ConvLayer(name="Conv1", c_in=256, c_out=256, kernel_shape=(3, 3), pads=(1, 1, 1, 1))
conv1_node = conv1.make_node(input_name=model_input_name, output_name="Conv1_Out")

relu1 = ReluLayer(name="ReLU1")
relu1_node = relu1.make_node(input_name="Conv1_Out", output_name="ReLU1_Out")

# node2
conv2 = ConvLayer(name="Conv2", c_in=256, c_out=256, kernel_shape=(1, 1), pads=(0, 0, 0, 0))
conv2_node = conv2.make_node(input_name=model_input_name, output_name="Conv2_Out")
 
relu2 = ReluLayer(name="ReLU2")
relu2_node = relu2.make_node(input_name="Conv2_Out", output_name="ReLU2_Out")

# node3
conv3 = ConvLayer(name="Conv3", c_in=256, c_out=256, kernel_shape=(3, 3), pads=(1, 1, 1, 1))
conv3_node = conv3.make_node(input_name="ReLU1_Out", output_name="Conv3_Out")

# node4
add1 = AddLayer(name="Add1")
add1_node = add1.make_node(input1_name="Conv3_Out", input2_name="ReLU2_Out", output_name="Add1_Out")

# node5
relu3 = ReluLayer(name="ReLU3")
relu3_node = relu3.make_node(input_name="Add1_Out", output_name=model_output_name)

# Create the graph
graph = helper.make_graph(
    nodes=[conv1_node, relu1_node, conv2_node, relu2_node, conv3_node, add1_node, relu3_node],
    name="Custom_Initial_Graph",
    inputs=[input_tensor],  # Graph input
    outputs=[output_tensor],  # Graph output
    initializer=[
        conv1.W_initializer_tensor, conv1.B_initializer_tensor,
        conv2.W_initializer_tensor, conv2.B_initializer_tensor,
        conv3.W_initializer_tensor, conv3.B_initializer_tensor
    ],
)

# Create and save the model
model = helper.make_model(graph, producer_name="onnx-custom-initial")
model.opset_import[0].version = 13

model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

onnx.save(model, "../../assets/onnx_files/initial_model.onnx")
