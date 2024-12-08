from onnx import ModelProto
import onnx
import numpy as np
import onnxruntime as ort
import time
from cost_model.graph_traversal import calculate_cost
import json
import torch
import os
import csv

def generate_dummy_input(session):
    inputs = session.get_inputs()
    dummy_input = {}
    default_dim = 10
    for input in inputs:
        shape = [dim if dim is not None and dim > 0 else default_dim for dim in input.shape]
        dummy_input[input.name] = np.random.random(shape).astype(np.float32)
    return dummy_input

def benchmark_model(model_path, num_runs=100, output_csv=None):
    
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    # sess_options = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    dummy_input = generate_dummy_input(session)

    for _ in range(10):
        session.run(None, dummy_input)

    times = []
    for _ in range(num_runs):
        start_time = time.time()
        session.run(None, dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # if output_csv:
    #     if os.path.dirname(output_csv):  # Ensure the directory is not empty
    #         os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    #     with open(output_csv, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Run", "Time (s)"])
    #         for i, t in enumerate(times, start=1):
    #             writer.writerow([i, t])


    avg_time = sum(times) / num_runs
    return avg_time

def benchmark(model1_path: str, model2_path: str, num_runs=100):
    time_model1 = time_model2 = 0
    time_model1 = benchmark_model(model1_path, num_runs)
    time_model2 = benchmark_model(model2_path, num_runs)
    return time_model1, time_model2

if __name__ == "__main__":
    # model_in_path = "assets/onnx_files/steps/model_step_1.onnx"
    model1_path = "assets/onnx_files/example_2_initial_model.onnx"
    model2_path = "assets/onnx_files/example_2_transform_1.onnx"
    model3_path = "assets/onnx_files/example_2_transform_2.onnx"
    model4_path = "assets/onnx_files/example_2_transform_3.onnx"


    mdl = onnx.load(model1_path)
    print(calculate_cost(mdl))
    mdl = onnx.load(model2_path)
    print(calculate_cost(mdl))
    mdl = onnx.load(model3_path)
    print(calculate_cost(mdl))
    mdl = onnx.load(model4_path)
    print(calculate_cost(mdl))

    time_model1 = benchmark_model(model1_path, 100, "model1.csv")
    print(f"Model 1 Average Time: {time_model1:.6f} seconds")
    time_model2 = benchmark_model(model2_path, 100, "model2.csv")
    print(f"Model 2 Average Time: {time_model2:.6f} seconds")
    time_model3 = benchmark_model(model3_path, 100, "model3.csv") 
    print(f"Model 3 Average Time: {time_model3:.6f} seconds")
    time_model4 = benchmark_model(model4_path, 100, "model4.csv")
    print(f"Model 4 Average Time: {time_model4:.6f} seconds")