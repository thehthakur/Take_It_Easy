import tf2onnx
import onnxruntime as rt
import onnx
import tensorflow as tf
import keras
import time
import numpy as np
import io
import os
import sys

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

def inference_time(onnx_model):
    if isinstance(onnx_model, str):
        session = rt.InferenceSession(onnx_model)
    elif isinstance(onnx_model, onnx.ModelProto):
        model_bytes = onnx_model.SerializeToString()
        session = rt.InferenceSession(io.BytesIO(model_bytes).read())
    else:
        raise ValueError("Input must be an ONNX model path or an onnx.ModelProto object")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # input_type = session.get_inputs()[0].type

    dummy_input = np.random.randn(1, *input_shape[1:]).astype(np.float32)

    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    start_time = time.time()
    outputs = session.run(None, {input_name: dummy_input})
    inference_time = time.time() - start_time

    print(f"Inference time: {inference_time:.6f} seconds")
    return inference_time


def main():
    model = keras.applications.MobileNetV2(weights='imagenet')
    model_input_shape = model.input_shape[1:]

    onnx_model_path = os.path.join(rootPath, "mobilenetv2.onnx")
    spec = (tf.TensorSpec((None, *model_input_shape), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_model_path)

    print(f"ONNX model saved to: {onnx_model_path}")

    inference_time(onnx_model_path)

    onnx_model = onnx.load(onnx_model_path)
    inference_time(onnx_model)


if __name__ == "__main__":
    main()