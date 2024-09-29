from onnx import helper

class AddLayer:
    def __init__(self, name):
        self.name = name
    
    def make_node(self, input1_name, input2_name, output_name):
        return helper.make_node(
            name=self.name,
            op_type="Add",
            inputs=[input1_name, input2_name],
            outputs=[output_name],
        )
