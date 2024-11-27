import os
import psutil
import subprocess
import re

# Get number of physical CPUs
NUM_CPUS = psutil.cpu_count(logical=False)

# Get cores per CPU
total_physical_cores = psutil.cpu_count(logical=False)
CORES_PER_CPU = total_physical_cores // NUM_CPUS if NUM_CPUS else total_physical_cores

# Get processing elements per core
logical_cores = os.cpu_count()
PES_PER_CORE = logical_cores // total_physical_cores if total_physical_cores else 1

# Estimate peak FLOPS
def get_peak_flops():
    # Get max clock speed (in Hz)
    if os.name == "posix":  # Linux/Mac
        clock_output = subprocess.check_output(["lscpu"], text=True)
        max_freq_match = re.search(r"CPU max MHz:\s+(\d+(\.\d+)?)", clock_output)
        max_freq = float(max_freq_match.group(1)) * 1e6 if max_freq_match else 2.5e9  # Default 2.5 GHz
    elif os.name == "nt":  # Windows
        clock_output = subprocess.check_output(["wmic", "cpu", "get", "MaxClockSpeed"], text=True)
        max_freq_match = re.search(r"\d+", clock_output)
        max_freq = float(max_freq_match.group(0)) * 1e6 if max_freq_match else 2.5e9

    # Approximate FLOPS calculation
    flops_per_cycle = 16  # Assume 16 FLOPs per cycle for modern CPUs
    peak_flops = max_freq * total_physical_cores * PES_PER_CORE * flops_per_cycle
    return peak_flops

PEAK_CPU_FLOPS = get_peak_flops()

# Output results
print(f"Number of CPUs: {NUM_CPUS}")
print(f"Cores per CPU: {CORES_PER_CPU}")
print(f"Processing Elements per Core: {PES_PER_CORE}")
print(f"Peak CPU FLOPS: {PEAK_CPU_FLOPS / 1e9:.2f} GFLOPS")
