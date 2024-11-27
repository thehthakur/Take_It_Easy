from onnx import ModelProto
import onnx
import numpy as np
import onnxruntime as ort
import time
from cost_model.graph_traversal import calculate_cost

def generate_dummy_input(session):
    inputs = session.get_inputs()
    dummy_input = {}
    for input in inputs:
        shape = [dim if dim > 0 else 1 for dim in input.shape]
        dummy_input[input.name] = np.random.random(shape).astype(np.float32)
    return dummy_input

def benchmark_model(model_path, num_runs=100):
    session = ort.InferenceSession(model_path)
    dummy_input = generate_dummy_input(session)

    for _ in range(10):
        session.run(None, dummy_input)

    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, dummy_input)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time

def benchmark(model1_path: str, model2_path: str, num_runs=100):
    time_model1 = benchmark_model(model1_path, num_runs)
    time_model2 = benchmark_model(model2_path, num_runs)
    return time_model1, time_model2

if __name__ == "__main__":
    model1_path = "assets/onnx_files/steps/model_step_0.onnx"
    model2_path = "assets/onnx_files/steps/model_step_2.onnx"

    # model = onnx.load(model1_path)
    # print(calculate_cost(model))
    # model = onnx.load(model2_path)
    # print(calculate_cost(model))

    time_model1, time_model2 = benchmark(model1_path, model2_path)

    print(f"Model 1 Average Time: {time_model1:.6f} seconds")
    print(f"Model 2 Average Time: {time_model2:.6f} seconds")
