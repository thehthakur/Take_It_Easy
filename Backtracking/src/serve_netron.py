import subprocess
import os
import sys

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

def serve_onnx_to_netron(path):
    try:
        subprocess.run(['netron', path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to open Netron: {e}")

# serve_onnx_to_netron(os.path.join(rootPath, "custom_graph.onnx"))