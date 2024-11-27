import numpy as np
import math

# attrlist contains the shape of the input
def add_mem(attrlist):
    shapes = np.array(attrlist)
    total_elements = np.prod(shapes)
    bytes_used = total_elements * 4
    return bytes_used

# attrlist contains the shape of the input
def relu_mem(attrlist):
    shapes = np.array(attrlist)
    total_elements = np.prod(shapes)
    bytes_used = total_elements * 4 
    return bytes_used

# attrlist contains a tuple (in_size, channels, kernel_size, filters, padding, stride)
def conv_mem(attrlist):
    xIn, channels, kernelSize, filters, padding, stride = attrlist
    
    xOut = math.floor((xIn + 2 * padding - kernelSize) / stride + 1)
    
    # Memory usage
    input_size = xIn * xIn * channels
    filter_size = kernelSize * kernelSize * channels * filters
    output_size = xOut * xOut * filters
    
    # Memory usage in bytes (assuming 4 bytes per element)
    bytes_used = (input_size + filter_size + output_size) * 4
    return bytes_used


def split_mem(attrlist):
    return 0