import onnx
import onnxoptimizer
import os
import sys
from serve_netron import serve_onnx_to_netron
from inference import inference_time
import heapq

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

passes = [
    "adjust_add",
    "rename_input_output",
    "set_unique_name_for_nodes",
    "nop",
    # "eliminate_nop_cast",
    # "eliminate_nop_dropout",
    # "eliminate_nop_flatten",
    # "extract_constant_to_initializer",
    # "eliminate_if_with_const_cond",
    # "eliminate_nop_monotone_argmax",
    # "eliminate_nop_pad",
    # "eliminate_nop_concat",
    # "eliminate_nop_split",
    # "eliminate_nop_expand",
    # "eliminate_shape_gather",
    # "eliminate_slice_after_shape",
    # "eliminate_nop_transpose",
    # "fuse_add_bias_into_conv",
    # "fuse_bn_into_conv",
    # "fuse_consecutive_concats",
    # "fuse_consecutive_log_softmax",
    # "fuse_consecutive_reduce_unsqueeze",
    # "fuse_consecutive_squeezes",
    # "fuse_consecutive_transposes",
    # "fuse_matmul_add_bias_into_gemm",
    # "fuse_pad_into_conv",
    # "fuse_pad_into_pool",
    # "fuse_transpose_into_gemm",
    # "replace_einsum_with_matmul",
    # # "lift_lexical_references",
    # # "split_init",
    # # "split_predict",
    # "fuse_concat_into_reshape",
    # "eliminate_nop_reshape",
    # "eliminate_nop_with_unit",
    # "eliminate_common_subexpression",
    # "fuse_qkv",
    # "fuse_consecutive_unsqueezes",
    # "eliminate_deadend",
    # "eliminate_identity",
    # "eliminate_shape_op",
    # "fuse_consecutive_slices",
    # "eliminate_unused_initializer",
    # "eliminate_duplicate_initializer",
    # "adjust_slice_and_matmul",
    # "rewrite_input_dtype",
]

def backtracking_search(onnx_model, alpha):
    pq = []

    heapq.heappush(pq, (inference_time(onnx_model), onnx_model))

    opt_onnx_model = onnx_model

    while len(pq):
        cost, onnx = heapq.heappop(pq)

        for _pass in passes:
            new_model = onnxoptimizer.optimize(onnx, [_pass])

            if inference_time(new_model) < inference_time(opt_onnx_model):
                opt_onnx_model = new_model

            if inference_time(new_model) < inference_time(opt_onnx_model) * alpha:
                heapq.heappush(pq, (inference_time(new_model), new_model))

    return opt_onnx_model


def main():
    onnx_model_path = os.path.join(rootPath, "substitution1_conv_graph.onnx")
    onnx_model_path_opt = os.path.join(rootPath, "substitution1_conv_graph_opt.onnx")
    onnx_model = onnx.load(onnx_model_path)

    # serve_onnx_to_netron(onnx_model_path)

    alpha = 1.5
    opt_onnx_model = backtracking_search(onnx_model, alpha)
    onnx.save(opt_onnx_model, onnx_model_path_opt)

if __name__ == "__main__":
    main()
    