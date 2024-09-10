import onnxoptimizer

optimizers = onnxoptimizer.get_available_passes()
[print(a) for a in optimizers]