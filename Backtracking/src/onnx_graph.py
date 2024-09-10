import onnx
import onnxoptimizer
import os
import sys
from serve_netron import serve_onnx_to_netron

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

def check_model(input_model_path):
    model = onnx.load(input_model_path)
    onnx.checker.check_model(model)
    print("Model is valid.")

def inspect_model(input_model_path):
    model = onnx.load(input_model_path)
    print("Model inputs:")
    for input in model.graph.input:
        print(f"- {input.name}")

    print("Model outputs:")
    for output in model.graph.output:
        print(f"- {output.name}")

    print("Model nodes:")
    for node in model.graph.node:
        print(f"- {node.op_type} ({node.name})")


def optimize_and_save_model(input_model_path, output_model_path):
    onnx_model = onnx.load(input_model_path)    
    all_passes = onnxoptimizer.get_available_passes()
    [print(f"\"{s}\",") for s in all_passes]
    all_passes.remove("lift_lexical_references")

    passes = [
        "adjust_add",
        "rename_input_output",
        "set_unique_name_for_nodes",
        "nop",
        "eliminate_nop_cast",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "extract_constant_to_initializer",
        "eliminate_if_with_const_cond",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_concat",
        "eliminate_nop_split",
        "eliminate_nop_expand",
        "eliminate_shape_gather",
        "eliminate_slice_after_shape",
        "eliminate_nop_transpose",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_pad_into_pool",
        "fuse_transpose_into_gemm",
        "replace_einsum_with_matmul",
        # "lift_lexical_references",
        # "split_init",
        # "split_predict",
        "fuse_concat_into_reshape",
        "eliminate_nop_reshape",
        "eliminate_nop_with_unit",
        "eliminate_common_subexpression",
        "fuse_qkv",
        "fuse_consecutive_unsqueezes",
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_shape_op",
        "fuse_consecutive_slices",
        "eliminate_unused_initializer",
        "eliminate_duplicate_initializer",
        "adjust_slice_and_matmul",
        "rewrite_input_dtype",
    ]

    # print(f"Applying the following optimization passes: {all_passes}")    
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    
    onnx.save(optimized_model, output_model_path)
    print(f"Optimized model has been saved to: {output_model_path}")

input_model_path = os.path.join(rootPath, 'custom_graph.onnx')
output_model_path = os.path.join(rootPath, 'custom_graph_optimized.onnx')

check_model(input_model_path)
# inspect_model(input_model_path)
optimize_and_save_model(input_model_path, output_model_path)

# serve_onnx_to_netron(os.path.join(rootPath, "mobilenetv2.onnx"))
serve_onnx_to_netron(output_model_path)