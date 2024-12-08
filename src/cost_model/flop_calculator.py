import numpy as np
import math

# attrlist contains the shape of the input
def add_flops(attrlist):
    shapes = np.array(attrlist)
    flops = np.prod(shapes)
    return flops

# attrlist contains the shape of the input
def relu_flops(attrlist):
    shapes = np.array(attrlist)
    flops = np.prod(shapes)
    return 2 * flops

# attrlist contains a tuple (in_size, channels, kernel_size, filters, padding, stride)
def conv_flops(attrlist):
    xIn, channels, kernelSize, filters, padding, stride = attrlist
    xOut = math.floor((xIn+2*padding-kernelSize)/stride + 1)
    flops = 2*(xOut**2)*(channels*kernelSize**2)*filters
    return flops

# Element-wise multiplication FLOPs
def mul_flops(attrlist):
    print(attrlist)
    shapes = np.array(attrlist)
    flops = np.prod(shapes)
    return flops

# Element-wise subtraction FLOPs
def sub_flops(attrlist):
    shapes = np.array(attrlist)
    flops = np.prod(shapes)
    return flops

def split_flops(attrlist):
    return 0