import numpy as np
import onnx
from onnx import helper

class ConvLayer:
    def __init__(self, name, c_in, c_out, kernel_shape, pads, strides=[1,1], weights=None, bias=None):
        self.name = name
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides

        # Initialize weights and bias if not provided
        self.W = weights if weights is not None else np.ones((c_out, c_in, *kernel_shape)).astype(np.float32)
        self.B = bias if bias is not None else np.ones((c_out,)).astype(np.float32)
        
        # Create initializer tensors for weights and bias
        self.W_initializer_tensor_name = f"{self.name}_W"
        self.W_initializer_tensor = self.create_initializer_tensor(self.W_initializer_tensor_name, self.W)

        self.B_initializer_tensor_name = f"{self.name}_B"
        self.B_initializer_tensor = self.create_initializer_tensor(self.B_initializer_tensor_name, self.B)
    
    def create_initializer_tensor(self, name, tensor_array):
        return onnx.helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=tensor_array.shape,
            vals=tensor_array.flatten(),
        )

    def make_node(self, input_name, output_name):
        return helper.make_node(
            name=self.name,
            op_type="Conv",
            inputs=[input_name, self.W_initializer_tensor_name, self.B_initializer_tensor_name],
            outputs=[output_name],
            kernel_shape=self.kernel_shape,
            pads=self.pads,
            strides=self.strides,
        )
