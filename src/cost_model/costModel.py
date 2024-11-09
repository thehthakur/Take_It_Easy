import numpy as np
import onnx
from onnx import numpy_helper
from typing import List
import onnxruntime as ort
import time
from onnx import helper, TensorProto
from typing import Dict
import math
import random
import re
import json

# TO DO:
# 1. attrList in PPP should have attributes in the same order as the ones that go to the make_node function for each operator. This will allow us to set up a default node-creation for all operators without having to write separate cases for all of them.

PEAK_CPU_FLOPS = 153.6e9*8
PPP_BENCH_OPS = ["conv2d"]

def returnFLOPs(op, attrList):
    print("     calculating FLOPs")
    match op:
        case "add":
            # attrList = [(tensor_dimensions_tuple)]
            if(len(attrList)!=1):
                print("wrong format for attribute list. Usage: ((tensor_dim0, tensor_dim1 ...))")
                exit(1)

            sum = 0
            for dim in attrList[0]:
                sum = sum*dim
            return sum
        case "conv2d":
            if(len(attrList)!=6):
                print("wrong format for attribute list. Usage: (in_size, channels, kernel_size, filters, padding, stride)")
                exit(1)

            # attrList = [in_size, channels, kernel_size, filters, padding, stride]
            xIn, channels, kernelSize, filters, padding, stride = attrList           
            xOut = math.floor((xIn+2*padding-kernelSize)/stride + 1)
            flops = 2*(xOut**2)*(channels*kernelSize**2)*filters
            return flops

        case "relu":
            if(len(attrList)!=1):
                print("wrong format for attribute list. Usage: (len_input_vector)")
                exit(1)

            return attrList[0]
        case "split":
            return 0 # just indexing is different and memory operations are reduced, which will be counted separate to the node's computation cost later.

def createONNXModel(operation: str, attrList) -> onnx.ModelProto:
    print("     creating onnx model")
    if operation == 'conv2d':
        xIn, channels, kernelSize, filters, padding, stride = attrList           

        # Declaring Input and Output Tensors (no values since they will be supplied to the model)
        # In0 ---->[ConvNode0]---> Out0 ---->[ConvNode1]---> Out1 --->[ConvNode2]---> Out2
        In0 = onnx.helper.make_tensor_value_info('In0', TensorProto.FLOAT, [1, channels, xIn, xIn]) 
        Out0 = onnx.helper.make_tensor_value_info('Out0', TensorProto.FLOAT, [1, channels, xIn, xIn])
        Out1 = onnx.helper.make_tensor_value_info('Out1', TensorProto.FLOAT, [1, filters, None, None])
        Out2 = onnx.helper.make_tensor_value_info('Out2', TensorProto.FLOAT, [1, None, None, None])

        # Declaring Default weights
        W0 = numpy_helper.from_array(
                    np.ones((channels, channels, 3, 3)).astype(np.float32), 
                    name='W0')

        B0 = numpy_helper.from_array(
                np.ones((channels)).astype(np.float32),
                name='B0')
        W1 = numpy_helper.from_array(
                    np.ones((filters, channels, kernelSize, kernelSize)).astype(np.float32), 
                    name='W1')

        B1 = numpy_helper.from_array(
                np.ones((filters)).astype(np.float32),
                name='B1')
        W2 = numpy_helper.from_array(
                    np.ones((32, filters, 1, 1)).astype(np.float32), 
                    name='W2')

        B2 = numpy_helper.from_array(
                np.ones((32)).astype(np.float32),
                name='B2')

        # Making Graph Nodes
        node0 = onnx.helper.make_node(name='Conv0', op_type='Conv', inputs=['In0', 'W0', 'B0'], outputs=['Out0'], strides=[1, 1], pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        node1 = onnx.helper.make_node(name='Conv1', op_type='Conv', inputs=['Out0', 'W1', 'B1'], outputs=['Out1'], strides=[stride, stride], pads=[padding, padding, padding, padding], kernel_shape=[kernelSize, kernelSize])

        node2 = onnx.helper.make_node(name='Conv2', op_type='Conv', inputs=['Out1', 'W2', 'B2'], outputs=['Out2'], strides=[1, 1], pads=[0, 0, 0, 0], kernel_shape=[1, 1])
        
        graph = onnx.helper.make_graph(
                nodes=[node0, node1, node2], 
                name='convBench', 
                inputs=[In0], 
                outputs=[Out2],
                initializer=[W0, B0, W1, B1, W2, B2])

        onnx_model = onnx.helper.make_model(graph)

        return onnx_model

        #convNode = onnx.helper.make_node(op_type="Conv", inputs=[main_input_name], outputs=[main_output_name], name="convNode", kernel_shape=)

def returnDefaultInputDict(op: str, attrList):
    print("     generating default input dict")
    match op:
        case "conv2d":
            inShape = [1, attrList[1], attrList[0], attrList[0]]
            inData = np.ones(inShape).astype(np.float32)
            inputs = {'In0' : inData}
            return inputs


def randAttrListGen(op):
    print("     generating attribute list")
    match op:
        case "conv2d":
            xIn = random.choice((14, 28, 56, 112, 224))
            kernelSize = random.choice((1, 3, 5, 7, 11))
            inputChannels = random.choice((2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096))
            filters = random.choice((2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536))
            stride = random.choice((1, 2))
            padding = random.choice((0, math.floor(kernelSize/2))) # either valid or same convolution
            return (xIn, inputChannels, kernelSize, filters, padding, stride)

        case "add":
            nDim = random.choice((2, 3))
            xIn = random.choice((14, 28, 56, 112, 224))
            attrList = ()
            for i in range (0, nDim):
                attrList.append(xIn)
            return attrList


def findPPP():
    print("finding PPP")
    avgPPP = 0
    benchCount = 0
    for op in PPP_BENCH_OPS:
        for i in range (0, 10):
            print(f"at benchcount = {i}, op = {op}")

            attrList = randAttrListGen(op)
            print(f"    attrList: {attrList}")
            opFLOPs = returnFLOPs(op, attrList)
            print(f"    FLOPs: {opFLOPs}")
            onnxModel = createONNXModel(op, attrList)
            onnx.save(onnxModel, 'modelBench.onnx')

            # Benchmark Start (using ONNX's built-in profiling tool) --------
            
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = True
            sess_options.intra_op_num_threads = 24
            session = ort.InferenceSession('modelBench.onnx', sess_options)
            inputDict = returnDefaultInputDict(op, attrList)
            
            print("about to run model")

            session.run(None, inputDict)

            print("running model")

            profileFile = session.end_profiling()
            with open(profileFile, 'r') as f:
                profileData = json.load(f)
            
            infTime = None
            match op:
                case "conv2d":
                    for entry in profileData:
                        if entry['name'] == 'Out1_nchwc_kernel_time':
                            infTime = entry['dur'] # duration in microseconds
                            break
            if infTime:
                print(f"Execution time for convolution: {infTime/1000.0} ms")
            else:
                print("inference time not found in profiling data")
                
            
            # Benchmark End ----------

            executionTimeActualSec = (infTime/1000000.0)
            executionTimeExpectedSec = (opFLOPs/PEAK_CPU_FLOPS)
            opPPP = executionTimeExpectedSec/executionTimeActualSec
            print(f"     opPPP: {opPPP}")

            avgPPP = (avgPPP*benchCount + opPPP)/(benchCount+1)
            benchCount = benchCount+1
    print(f"Average PPP: {avgPPP}")


estimatePPP = findPPP()


