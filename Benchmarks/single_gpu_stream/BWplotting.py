import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import sys

# Load data from file
data = np.genfromtxt(sys.argv[1], skip_header=1)

# Extract columns
datasizes = data[:, 0]
h2d = data[:, 1]
d2h = data[:, 2]
kernel = data[:, 3]

# Group by datasize to compute averages
grouped = defaultdict(list)
for size, h, d, k in zip(datasizes, h2d, d2h, kernel):
    grouped[size].append((h, d, k))

# Sort datasizes and compute means
sorted_sizes = sorted(grouped.keys())
h2d_avg, d2h_avg, kernel_avg = [], [], []

for size in sorted_sizes:
    values = np.array(grouped[size])
    h2d_avg.append(np.mean(values[:, 0]))
    d2h_avg.append(np.mean(values[:, 1]))
    kernel_avg.append(np.mean(values[:, 2]))

# Convert to NumPy arrays
sorted_sizes = np.array(sorted_sizes)
h2d_avg = np.array(h2d_avg)
d2h_avg = np.array(d2h_avg)
kernel_avg = np.array(kernel_avg)

# Plot configs
bandwidths = [
    ('H2DBW', h2d_avg, 'h2dbw.png'),
    ('D2HBW', d2h_avg, 'd2hbw.png'),
    ('KernelBW', kernel_avg, 'kernelbw.png')
]

for name, values, filename in bandwidths:
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_sizes, values, marker='o')
    plt.xscale('log', base=1.2)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)} B'))
    plt.xlabel('Data Size (bytes)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title(f'{name} vs Data Size (log₁.₂ scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close figure to prevent display
