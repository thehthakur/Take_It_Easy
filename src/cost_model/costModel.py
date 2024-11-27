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
# from costEstimation import get_flops
from cost_model.flop_calculator import *

# TO DO:
# 1. attrList in PPP should have attributes in the same order as the ones that go to the make_node function for each operator. This will allow us to set up a default node-creation for all operators without having to write separate cases for all of them.
# 2. write an estimateOpCost(node: NodeProto) function
NUM_CPUS=1
CORES_PER_CPU=8
PES_PER_CORE=1
PEAK_CPU_FLOPS = 256e9
PPP_BENCH_OPS = ["Relu", "Conv", "Add"]

def returnFLOPs(op, attrlist):
    print("[INFO] calculating FLOPs")
    match op:
        case "Add":
            return add_flops(attrlist)
        case "Conv":
            return conv_flops(attrlist)
        case "Relu":
            return relu_flops(attrlist)
        case "Split":
            return split_flops(attrlist)
    print("Assuming {op} to be zero flops")
    return 0

def createONNXModel(operation: str, attrList) -> onnx.ModelProto:
    print("[INFO] creating onnx model")
    if operation == 'Conv':
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
    elif operation == 'Add':
        matrix = attrList
        In00 = onnx.helper.make_tensor_value_info('In00', TensorProto.FLOAT, matrix)
        In01 = onnx.helper.make_tensor_value_info('In01', TensorProto.FLOAT, matrix)
        Out00 = onnx.helper.make_tensor_value_info('Out00', TensorProto.FLOAT, matrix)
        Out10 = onnx.helper.make_tensor_value_info('Out10', TensorProto.FLOAT, matrix)
        Out20 = onnx.helper.make_tensor_value_info('Out20', TensorProto.FLOAT, matrix)

        node0 = onnx.helper.make_node(name='Add0', op_type='Add', inputs=['In00', 'In01'], outputs=['Out00'])
        node1 = onnx.helper.make_node(name='Add1', op_type='Add', inputs=['Out00', 'In01'], outputs=['Out10'])
        node2 = onnx.helper.make_node(name='Add2', op_type='Add', inputs=['Out10', 'In01'], outputs=['Out20'])

        graph = onnx.helper.make_graph(
                nodes=[node0, node1, node2],
                name='addBench',
                inputs=[In00, In01],
                outputs=[Out20])

        onnx_model = onnx.helper.make_model(graph)
        return onnx_model

    elif operation == 'Relu':
        matrix = attrList
        In0 = onnx.helper.make_tensor_value_info('In0', TensorProto.FLOAT, matrix)
        Out0 = onnx.helper.make_tensor_value_info('Out0', TensorProto.FLOAT, matrix)
        Out1 = onnx.helper.make_tensor_value_info('Out1', TensorProto.FLOAT, matrix)
        Out2 = onnx.helper.make_tensor_value_info('Out2', TensorProto.FLOAT, matrix)

        node0 = onnx.helper.make_node(name='Relu0', op_type='Relu', inputs=['In0'], outputs=['Out0'])
        node1 = onnx.helper.make_node(name='Relu1', op_type='Relu', inputs=['Out0'], outputs=['Out1'])
        node2 = onnx.helper.make_node(name='Relu2', op_type='Relu', inputs=['Out1'], outputs=['Out2'])

        graph = onnx.helper.make_graph(
                nodes=[node0, node1, node2],
                name='reluBench',
                inputs=[In0],
                outputs=[Out2])

        onnx_model = onnx.helper.make_model(graph)
        return onnx_model

        #convNode = onnx.helper.make_node(op_type="Conv", inputs=[main_input_name], outputs=[main_output_name], name="convNode", kernel_shape=)

def returnDefaultInputDict(op: str, attrList):
    print("[INFO] Generating default input dict")
    match op:
        case "Conv":
            inShape = [1, attrList[1], attrList[0], attrList[0]]
            inData = np.ones(inShape).astype(np.float32)
            inputs = {'In0' : inData}
            return inputs
        case "Add":
            inData = np.ones(attrList).astype(np.float32)
            inputs = {'In00': inData, 'In01': inData}
            return inputs
        case "Relu":
            inData = np.ones(attrList).astype(np.float32)
            inputs = {'In0': inData}
            return inputs

def randAttrListGen(op):
    print("[INFO] Generating Attribute List")
    match op:
        case "Conv":
            xIn = random.choice((14, 28, 56, 112, 224))
            kernelSize = random.choice((1, 3, 5, 7, 11))
            inputChannels = random.choice((2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096))
            filters = random.choice((2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536))
            stride = random.choice((1, 2))
            padding = random.choice((0, math.floor(kernelSize/2)))
            return (xIn, inputChannels, kernelSize, filters, padding, stride)
        case "Add":
            nDim = random.choice((2, 3))
            attrList = ()
            for dim in range(nDim):
                xIn = random.choice((14, 28, 56, 112, 224))
                attrList = attrList + (xIn, )
            return attrList
        case "Relu":
            nDim = random.choice((2, 3))
            attrList = ()
            for dim in range(nDim):
                xIn = random.choice((14, 28, 56, 112, 224))
                attrList = attrList + (xIn, )
            return attrList
        
def findPPP():
    print("finding PPP")
    PPPDict = {}
    numOfIterations = 15
    for op in PPP_BENCH_OPS:
        avgPPP = 0
        benchCount = 0

        for i in range (0, numOfIterations):
            print(f"at benchcount = {i}, op = {op}")

            attrList = randAttrListGen(op)
            print(f"    attrList: {attrList}")
            opFLOPs = returnFLOPs(op, attrList)
            print(f"    FLOPs: {opFLOPs}")
            onnxModel = createONNXModel(op, attrList)
            new_opset_version = 21
            for opset in onnxModel.opset_import:
                opset.version = new_opset_version

            onnx.save(onnxModel, 'modelBench.onnx')

            # Benchmark Start (using ONNX's built-in profiling tool) --------
            
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.intra_op_num_threads = 8 #NUM_CPUS*CORES_PER_CPU*PES_PER_CORE
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession('modelBench.onnx', sess_options)
            inputDict = returnDefaultInputDict(op, attrList)
            
            print("     about to run model")

            session.run(None, inputDict)

            print("     running model")

            profileFile = session.end_profiling()
            with open(profileFile, 'r') as f:
                profileData = json.load(f)
            
            infTime = None
            match op:
                case "Conv":
                    for entry in profileData:
                        if entry['name'] == 'Conv1_kernel_time':
                            infTime = entry['dur'] # duration in microseconds
                            break
                case "Add":
                    for entry in profileData:
                        if entry['name'] == 'Add1_kernel_time':
                            infTime = entry['dur'] 
                            break
                case "Relu":
                    for entry in profileData:
                        if entry['name'] == 'Relu1_kernel_time':
                            infTime = entry['dur']
                            break
            if infTime:
                print(f"    Execution time for convolution: {infTime/1000.0} ms")
            else:
                print("     inference time not found in profiling data")
                
            
            # Benchmark End ----------
            opPPP = -1
            if(infTime!=None):
                executionTimeActualSec = (infTime/1000000.0)
                executionTimeExpectedSec = (opFLOPs/PEAK_CPU_FLOPS)
                print(f"    expected execution time: {executionTimeExpectedSec*1000} ms")
                print(f"    Flops calculated {opFLOPs}")
                opPPP = executionTimeExpectedSec/executionTimeActualSec
            print(f"     opPPP: {opPPP}")

            avgPPP = (avgPPP*benchCount + opPPP)/(benchCount+1)
            benchCount = benchCount+1

        PPPDict[op] = avgPPP
        print(f"    Average PPP for {op}: {avgPPP}")
    return PPPDict

def estimateOpCost(node):
    match node.op_type:
        case "Conv":
            return 100
        case "Add":
            return 20
        case "Relu":
            return 10
        case "Split":
            return 0
    return 100.0

def getSyncBarriers(G: onnx.GraphProto):
    # returns all nodes which have multiple input nodes
    syncBarrierNodeList = []
    # for all nodes having more than one prerequisite-nodes, 
    # add their name to the list of synchronization barriers.
    for n in G.node:
        numInputs = 0
        for tensor in n.input:
            # Check if any node in the graph has the tensor in its output
            count = sum(1 for M in graph.node if tensor in M.output)
            if count > 0:
                numInputs += 1

        if(numInputs>1):
            syncBarrierNodeList.append(n.name)
    return syncBarrierNodeList
            
def getBranches(G: onnx.GraphProto):
    # returns all nodes which have multiple output nodes
    branchNodeList = []
    # for all nodes having more than one prerequisite-nodes, 
    # add their name to the list of synchronization barriers.
    for n in G.node:
        numOutputs = 0
        for tensor in n.output:
            # Check if any node in the graph has the tensor in its output
            count = sum(1 for M in graph.node if tensor in M.input)
            if count > 0:
                numOutputs += 1

        if(numOutputs>1):
            branchNodeList.append(n.name)
    return branchNodeList

def getSuccessors(G: onnx.GraphProto, N: onnx.NodeProto):
    succList = []
    for tensor in N.output:
        for node in G.node:
            if (tensor in node.input) and (node not in succList):
                succList.append(node)
    return succList

def processFromNode(curNode: onnx.NodeProto, syncBarriers, branches, syncReduce, visited):
    subGraphCost = 0
    if curNode.name not in visited:
        visited.append(curNode.name)

    subGraphCost += estimateOpCost(curNode)

    if curNode.name in branches:
        # totalCostForAllBranches
        for node in getSuccessors(curNode):
            subGraphCost = max(subGraphCost)
    # initialize cost to current node's cost
    # then keep adding cost for next nodes until we reach either a branch or a sync barrier
    # the goal is to, in this way, calculate costs for all sequential parts of our graph,
    # which can be later combined to form a whole
    while True:
        # break either if current node has more than one successors, or if some successor
        # has a synchronization barrier
        if(curNode.name in branches):
            break
        succ = getSuccessors(curNode)
        count = sum(1 for M in succ if M.name in syncBarriers)
        if(count>0):
            # if next node has a synchronization barrier,
            # collect the cost of this serial thread onto that node
            # so that it can be reduced later with other branches reaching that sync barrier
            for successor in succ:
                if successor.name in syncBarriers:
                    syncReduce[successor.name].append(serialCost)
            break

        curNode = succ[0]
        visited.append(curNode.name)
        serialCost += estimateOpCost(curNode)
    
def estimateGraphCost(G: onnx.GraphProto):
    syncBarriers = getSyncBarriers(G)
    branches = getBranches(G)
    syncReduce = {}
    for nSync in syncBarriers:
        syncReduce[nSync] = 0
    visited = []
    allNodes= G.node
    curNode = allNodes[0]
    while len(visited) != len(allNodes):
        for node in allNodes:
            if node not in visited:
                curNode = node
                visited.append(curNode.name)
                serialCost = estimateOpCost(curNode)
                # initialize cost to current node's cost
                # then keep adding cost for next nodes until we reach either a branch or a sync barrier
                # the goal is to, in this way, calculate costs for all sequential parts of our graph,
                # which can be later combined to form a whole
                while True:
                    # break either if current node has more than one successors, or if some successor
                    # has a synchronization barrier
                    if(curNode.name in branches):
                        break
                    succ = getSuccessors(curNode)
                    count = sum(1 for M in succ if M.name in syncBarriers)
                    if(count>0):
                        # if next node has a synchronization barrier,
                        # collect the cost of this serial thread onto that node
                        # so that it can be reduced later with other branches reaching that sync barrier
                        for successor in succ:
                            if successor.name in syncBarriers:
                                syncReduce[successor.name].append(serialCost)
                        break

                    curNode = succ[0]
                    visited.append(curNode.name)
                    serialCost += estimateOpCost(curNode)
                    
if __name__ == "__main__":
    estimatePPP = findPPP()
    print(f"PPPs for various operations: {estimatePPP}")
    model = onnx.load("../../assets/onnx_files/example_1_initial_model.onnx")
    graph = model.graph
    l = getSyncBarriers(graph)
    print(f"list of sync barriers: {l}")
