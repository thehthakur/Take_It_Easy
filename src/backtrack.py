import onnx
from onnx import ModelProto
import sys
import heapq
import copy

from utils import get_inference_time, apply_operation, render_onnx_with_netron, combine_images_with_annotations
from graph_transforms.enlarge_conv_kernel import enlarge_conv_kernel
from graph_transforms.fuse_conv_layers import fuse_conv_layers

# Backtracking with optional visualization
def backtrack(model, cost_model, alpha, operations, visualize=True, output_image="path_timeline.png"):
    queue = []
    heapq.heappush(queue, (cost_model(model), copy.deepcopy(model)))  # Use deepcopy here
    optimized_model = copy.deepcopy(model)  # Ensure optimized_model is also a deepcopy
    optimized_mdl_cost = float(cost_model(optimized_model))
    path = []
    step = 0
    onnx_files_directory = "../assets/onnx_files"
    img_files_directory = "../assets/images"
    onnx.save_model(model, f"{onnx_files_directory}/steps/model_step_{step}.onnx")

    while queue and step < 3:
        _, curr_model = heapq.heappop(queue)
        curr_model = copy.deepcopy(curr_model)  # Ensure curr_model is a deepcopy
        for operation in operations:
            new_models = apply_operation(curr_model, operation)
            if len(new_models) == 0:
                continue
            for mdl in new_models:
                mdl = copy.deepcopy(mdl)  # Ensure each mdl is a deepcopy
                mdl_cost = float(cost_model(mdl))
                if mdl_cost < float(alpha * optimized_mdl_cost):
                    heapq.heappush(queue, (mdl_cost, mdl))
                    step += 1
                    onnx.save_model(mdl, f"{onnx_files_directory}/steps/model_step_{step}.onnx")
                    path.append((f"model_step_{step - 1}.onnx", f"model_step_{step}.onnx", operation.__name__))

                    if mdl_cost < optimized_mdl_cost:
                        optimized_model = mdl
                        optimized_mdl_cost = mdl_cost

    # Render ONNX models and combine into visualization if requested
    if visualize:
        rendered_images = []
        annotations = []
        
        # Include the original model as the first image in the path
        render_onnx_with_netron(f"{onnx_files_directory}/steps/model_step_0.onnx", f"{img_files_directory}/steps/step_0.png")
        rendered_images.append("step_0.png")
        annotations.append("Original Model")

        # Render subsequent steps and annotate
        for step, (from_model, to_model, operation) in enumerate(path):
            render_onnx_with_netron(to_model, f"{img_files_directory}/steps/step_{step + 1}.png")
            rendered_images.append(f"step_{step + 1}.png")
            annotations.append(f"Step {step + 1}: {operation}")
        
        # Combine all images with annotations into the final visualization
        combine_images_with_annotations(rendered_images, annotations, f"{img_files_directory}/{output_image}")
    
    return optimized_model

if __name__ == "__main__":
    def cost_model(m: ModelProto):
        # return calculate_cost(m)
        return get_inference_time(m)  # Replace later with actual cost model

    model_path = "../assets/onnx_files/example_1_initial_model.onnx"
    model = onnx.load_model(model_path)
    alpha = 1.5
    operations = [enlarge_conv_kernel, fuse_conv_layers]

    optimized_model = backtrack(
        model=model,
        cost_model=cost_model,
        alpha=alpha,
        operations=operations,
        visualize=False,
        output_image="optimization_steps.png"
    )
    onnx.save(optimized_model, "../assets/onnx_files/optimized_model.onnx")
