import onnx
from onnx import ModelProto
import onnxruntime as rt
import onnx
import time
import numpy as np
import io
import subprocess
from selenium import webdriver
from PIL import Image, ImageDraw, ImageFont


def get_adjacency_list(model: ModelProto):
    adjacency_list = {}

    for input_tensor in model.graph.input:
        input_name = input_tensor.name
        if input_name not in adjacency_list:
            adjacency_list[input_name] = []

        for node in model.graph.node:
            if input_name in node.input:
                node_name = node.name if node.name else node.output[0]
                adjacency_list[input_name].append(node_name)

    for node in model.graph.node:
        node_name = node.name if node.name else node.output[0]
        if node_name not in adjacency_list:
            adjacency_list[node_name] = []

        for output in node.output:
            for other_node in model.graph.node:
                if output in other_node.input:
                    connected_node_name = other_node.name if other_node.name else other_node.output[0]
                    adjacency_list[node_name].append(connected_node_name)

    for output_tensor in model.graph.output:
        output_name = output_tensor.name
        if output_name not in adjacency_list:
            adjacency_list[output_name] = []

        for node in model.graph.node:
            if output_name in node.output:
                node_name = node.name if node.name else node.output[0]
                adjacency_list[node_name].append(output_name)

    return adjacency_list

def get_convs_with_shared_parent(adjacency_list, opName_to_opType):
    conv_pairs = []  # To store pairs of Conv layers sharing the same parent

    for parent, children in adjacency_list.items():
        # Filter children that are of type "Conv"
        conv_layers = [child for child in children if opName_to_opType.get(child) == "Conv"]

        # If two or more Conv layers share the same parent
        if len(conv_layers) > 1:
            # Create all possible pairs of Conv layers
            for i in range(len(conv_layers)):
                for j in range(i + 1, len(conv_layers)):
                    conv_pairs.append((conv_layers[i], conv_layers[j]))

    return conv_pairs

def apply_operation(model: ModelProto, operation):
    new_models = []
    operation_name = operation.__name__
    
        
    if operation_name == "enlarge_conv_kernel":
        enlarged_kernel_size = [3,3]
        graph = model.graph
        
        for node in graph.node:
            if node.op_type == "Conv":
                new_model = operation(model, node.name, enlarged_kernel_size)
                if new_model == -1:
                    continue
                else:
                    new_models.append(new_model)
        return new_models
    elif operation_name == "fuse_conv_layers":
        adjacency_list = get_adjacency_list(model)
        opName_to_opType = {}
        graph = model.graph
        
        for node in graph.node:
            opName_to_opType[node.name] = node.op_type
        conv_pairs = get_convs_with_shared_parent(adjacency_list, opName_to_opType)
        
        for conv_pair in conv_pairs:
            new_model = operation(model, conv_pair[0], conv_pair[1])
            new_models.append(new_model)
        return new_models
    
def get_inference_time(onnx_model):
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
    return inference_time

def render_onnx_with_netron(onnx_path, output_image_path, netron_port=8081):
    # Start Netron server
    netron_process = subprocess.Popen(
        ["netron", "--host", "localhost", "--port", str(netron_port), onnx_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(5)  # Allow Netron to start
        # Set up Selenium to capture the rendered graph
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=options)
        driver.get(f"http://localhost:{netron_port}")
        time.sleep(5)  # Allow time for rendering
        driver.save_screenshot(output_image_path)
        driver.quit()
        # Crop image to remove browser UI
        image = Image.open(output_image_path)
        cropped_image = image.crop((50, 50, image.width - 50, image.height))
        cropped_image.save(output_image_path)
    finally:
        netron_process.terminate()

# Combine images with annotations
def combine_images_with_annotations(image_paths, annotations, output_image_path):
    images = [Image.open(img_path) for img_path in image_paths]
    total_width = sum(img.width for img in images) + (len(images) - 1) * 50
    max_height = max(img.height for img in images)
    combined_image = Image.new("RGB", (total_width, max_height + 100), "white")
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.load_default()
    x_offset = 0
    for i, img in enumerate(images):
        combined_image.paste(img, (x_offset, 50))
        draw.text((x_offset + img.width // 2, 20), annotations[i], fill="black", font=font)
        x_offset += img.width + 50
    combined_image.save(output_image_path)