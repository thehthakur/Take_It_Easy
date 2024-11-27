import numpy as np
import onnx
from onnx import numpy_helper
from typing import List

def enlarge_conv_kernel(model: onnx.onnx_ml_pb2.ModelProto, node_to_modify: str, enlarged_kernel_size: List[int]) -> onnx.onnx_ml_pb2.ModelProto:
    graph = model.graph
    
    # Step 1: Find the Conv node and modify its kernel size attribute
    for node in graph.node:
        if node.name == node_to_modify:
            for attribute in node.attribute:
                if attribute.name == 'kernel_shape':
                    if attribute.ints[:] == enlarged_kernel_size:
                        return -1
                    print(f"Original kernel shape: {attribute.ints}")
                    attribute.ints[:] = enlarged_kernel_size  # Update kernel shape
                    print(f"Updated kernel shape: {attribute.ints}")
                elif attribute.name == 'pads':
                    attribute.ints[:] = [1, 1, 1, 1]

    # Step 2: Adjust the initializer (weights) corresponding to the Conv layer
    for initializer in graph.initializer:
        if initializer.name == f"{node_to_modify}_W":
            # Convert the initializer to a NumPy array
            weights = numpy_helper.to_array(initializer)
            print(f"Original weight shape: {weights.shape}")

            # Calculate new weight shape based on the enlarged kernel size
            new_weight_shape = (
                weights.shape[0],  # Number of output channels (remains unchanged)
                weights.shape[1],  # Number of input channels (remains unchanged)
                enlarged_kernel_size[0],  # New kernel height
                enlarged_kernel_size[1],  # New kernel width
            )

            # Create new weights with adjusted shape (filled with random values as a placeholder)
            # You may want to handle how these values are initialized
            new_weights = np.ones(new_weight_shape).astype(np.float32)

            # Update the initializer with new weights
            new_initializer = numpy_helper.from_array(new_weights, initializer.name)
            graph.initializer.remove(initializer)  # Remove old initializer
            graph.initializer.append(new_initializer)  # Add updated initializer

            print(f"Updated weight shape: {new_weights.shape}")
            break
    return model

# example usage
# model = onnx.load("../../assets/onnx_files/example_1_initial_model.onnx")
# transformed_model = enlarge_conv_kernel(model=model, node_to_modify="Conv2", enlarged_kernel_size=[3, 3]) 
