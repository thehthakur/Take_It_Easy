import subprocess
import os
import sys
import argparse

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

def serve_onnx_to_netron(path):
    try:
        subprocess.run(['netron', path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to open Netron: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an ONNX model to Netron.")
    parser.add_argument("path", type=str, help="Path to the ONNX model file from root folder")
    args = parser.parse_args()
    model_path = os.path.join(rootPath, args.path)
    serve_onnx_to_netron(model_path)
    # print(model_path)