import onnx
import onnx.helper as helper
from onnx import TensorProto

x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 224, 224])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 224, 224])
z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 3, 224, 224])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

node1 = helper.make_node('Mul', name="node1", inputs=['x', 'y'], outputs=['x_mul_y'])
node2 = helper.make_node('Sub', name="node2", inputs=['one', 'x'], outputs=['one_minus_x'])
node3 = helper.make_node('Mul', name="node3", inputs=['one_minus_x', 'z'], outputs=['one_minus_x_mul_z'])
node4 = helper.make_node('Add', name="node4", inputs=['x_mul_y', 'one_minus_x_mul_z'], outputs=['output'])

graph = helper.make_graph(
    nodes=[node1, node2, node3, node4],
    name='GraphWith4OperatorsFirstEquation',
    inputs=[x, y, z],
    outputs=[output],
    initializer=[
        helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])
    ]
)

model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[onnx.helper.make_opsetid("", 21)])
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, 'assets/onnx_files/example_2_initial_model.onnx')

x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 224, 224])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 224, 224])
z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 3, 224, 224])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

node1 = helper.make_node('Mul', name="node1", inputs=['x', 'y'], outputs=['x_mul_y'])
node2 = helper.make_node('Mul', name="node2", inputs=['x', 'z'], outputs=['x_mul_z_1'])
node3 = helper.make_node('Add', name="node3", inputs=['x_mul_y', 'x_mul_z_1'], outputs=['x_mul_y_plus_x_mul_z'])
node4 = helper.make_node('Mul', name="node4", inputs=['x', 'z'], outputs=['x_mul_z_2'])
node5 = helper.make_node('Sub', name="node5", inputs=['x_mul_y_plus_x_mul_z', 'x_mul_z_2'], outputs=['output'])

graph = helper.make_graph(
    nodes=[node1, node2, node3, node4, node5],
    name='GraphWith5OperatorsSecondEquation',
    inputs=[x, y, z],
    outputs=[output]
)

model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[onnx.helper.make_opsetid("", 21)])
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, 'assets/onnx_files/example_2_transform_1.onnx')

x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 224, 224])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 224, 224])
z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 3, 224, 224])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

node1 = helper.make_node('Mul', name="node1", inputs=['x', 'y'], outputs=['x_mul_y'])
node2 = helper.make_node('Mul', name="node2", inputs=['x', 'z'], outputs=['x_mul_z'])
node3 = helper.make_node('Sub', name="node3", inputs=['x_mul_y', 'x_mul_z'], outputs=['x_mul_y_minus_x_mul_z'])
node4 = helper.make_node('Add', name="node4", inputs=['x_mul_y_minus_x_mul_z', 'z'], outputs=['output'])

graph = helper.make_graph(
    nodes=[node1, node2, node3, node4],
    name='GraphWith4Operators',
    inputs=[x, y, z],
    outputs=[output]
)

model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[onnx.helper.make_opsetid("", 21)])
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, 'assets/onnx_files/example_2_transform_2.onnx')

x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 224, 224])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 224, 224])
z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 3, 224, 224])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

node1 = helper.make_node('Sub', name="node1", inputs=['y', 'z'], outputs=['y_minus_z'])
node2 = helper.make_node('Mul', name="node2", inputs=['x', 'y_minus_z'], outputs=['x_mul_y_minus_z'])
node3 = helper.make_node('Add', name="node3", inputs=['x_mul_y_minus_z', 'z'], outputs=['output'])

graph = helper.make_graph(
    nodes=[node1, node2, node3],
    name='GraphWith3OperatorsFinalEquation',
    inputs=[x, y, z],
    outputs=[output]
)

model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[onnx.helper.make_opsetid("", 21)])
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, 'assets/onnx_files/example_2_transform_3.onnx')