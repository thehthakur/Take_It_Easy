from onnx import helper

class ReluLayer:
    def __init__(self, name):
        self.name = name
    
    def make_node(self, input_name, output_name):
        return helper.make_node(
            name=self.name,
            op_type="Relu",
            inputs=[input_name],
            outputs=[output_name],
        )
