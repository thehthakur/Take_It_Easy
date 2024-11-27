import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import List


def check_same_padding_stride_kernel(conv1, conv2):
    conv1_pads = [attr.ints for attr in conv1.attribute if attr.name == "pads"][0]
    conv2_pads = [attr.ints for attr in conv2.attribute if attr.name == "pads"][0]
    
    conv1_strides = [attr.ints for attr in conv1.attribute if attr.name == "strides"][0]
    conv2_strides = [attr.ints for attr in conv2.attribute if attr.name == "strides"][0]
    
    conv1_kernel = [attr.ints for attr in conv1.attribute if attr.name == "kernel_shape"][0]
    conv2_kernel = [attr.ints for attr in conv2.attribute if attr.name == "kernel_shape"][0]
    
    assert conv1_pads == conv2_pads, "Padding is not the same between the two layers!"
    assert conv1_strides == conv2_strides, "Stride is not the same between the two layers!"
    assert conv1_kernel == conv2_kernel, "Kernel size is not the same between the two layers!"
    return True

def find_direct_child_node(graph, conv_output):
    # Iterate through all nodes in the graph
    for node in graph.node:
        # Check if the node has conv_output as one of its inputs
        if conv_output in node.input:
            return node
    return None

def fuse_conv_layers(model, conv1_name, conv2_name):
    graph = model.graph

    conv1_node = None
    conv2_node = None

    # Find the convolution layers in the model by name
    for node in graph.node:
        if node.name == conv1_name:
            conv1_node = node
        if node.name == conv2_name:
            conv2_node = node

    assert conv1_node is not None and conv2_node is not None, "Convolution layers not found in the graph!"

    # Check if padding, stride, and kernel sizes are the same
    check_same_padding_stride_kernel(conv1_node, conv2_node)

    # Get the outputs of Conv1 and Conv2
    conv1_output = conv1_node.output[0]
    conv2_output = conv2_node.output[0]

    # Find the direct child nodes of Conv1 and Conv2
    conv1_child = find_direct_child_node(graph, conv1_output)
    conv2_child = find_direct_child_node(graph, conv2_output)

    # Ensure both child nodes are not None
    assert conv1_child is not None, f"No direct child found for Conv1 ({conv1_name})"
    assert conv2_child is not None, f"No direct child found for Conv2 ({conv2_name})"

    print(f"Conv1 Child: {conv1_child.name}")
    print(f"Conv2 Child: {conv2_child.name}")

    # Get the weights and biases for the two layers
    conv1_w = next(init for init in graph.initializer if init.name == conv1_node.input[1])
    conv1_b = next(init for init in graph.initializer if init.name == conv1_node.input[2])

    conv2_w = next(init for init in graph.initializer if init.name == conv2_node.input[1])
    conv2_b = next(init for init in graph.initializer if init.name == conv2_node.input[2])

    # Convert the ONNX tensors to NumPy arrays
    conv1_w_np = numpy_helper.to_array(conv1_w)
    conv1_b_np = numpy_helper.to_array(conv1_b)
    
    conv2_w_np = numpy_helper.to_array(conv2_w)
    conv2_b_np = numpy_helper.to_array(conv2_b)

    # Merge weights: Concatenate along the output channel dimension (axis 0)
    merged_w_np = np.concatenate((conv1_w_np, conv2_w_np), axis=0)
    merged_b_np = np.concatenate((conv1_b_np, conv2_b_np), axis=0)

    # Update conv1 weights and biases to the merged version
    conv1_w.CopyFrom(numpy_helper.from_array(merged_w_np, conv1_w.name))
    conv1_b.CopyFrom(numpy_helper.from_array(merged_b_np, conv1_b.name))

    # Modify the output shape of Conv1 to account for the increased channels (512 instead of 256)
    conv1_node.output[0] = "Conv1_Fused_Out"

    for node in graph.node:
        # Update nodes expecting Conv1_Out to use Conv1_Split1_Out
        for i, input_name in enumerate(node.input):
            if input_name == conv1_output:  # Replace old Conv1 output
                node.input[i] = 'Conv1_Split1_Out'
    
        # Update nodes expecting Conv2_Out to use Conv1_Split2_Out
        for i, input_name in enumerate(node.input):
            if input_name == conv2_output:  # Replace old Conv2 output
                node.input[i] = 'Conv1_Split2_Out'

    # Create a split node to split the output into two 256-channel tensors
    split_node = helper.make_node(
        'Split',
        inputs=['Conv1_Fused_Out'],
        outputs=['Conv1_Split1_Out', 'Conv1_Split2_Out'],
        axis=1,  # Splitting along the channel axis
        name="Split1"
    )

    # Add the new split node to the graph
    graph.node.extend([split_node])

    print(conv1_child.input)
    print(conv1_child.output)
    print(conv2_child.input)
    print(conv2_child.output)
    
    
    # Remove the second convolution layer (conv2)
    graph.node.remove(conv2_node)

    return model

# example usage
if __name__ == '__main__':
    # Example usage:
    model_path = "../../assets/onnx_files/example_1_transformation_1_enlarge_conv_kernel.onnx"
    conv1_name = "Conv1"
    conv2_name = "Conv2"
    fused_model = fuse_conv_layers(model_path, conv1_name, conv2_name)
