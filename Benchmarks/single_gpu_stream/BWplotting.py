import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys


def main():
    # Load data from file
    datafilename = sys.argv[1]   
    data = np.genfromtxt(datafilename, skip_header=1)

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
    sorted_sizes = np.array(sorted_sizes) / 1e9
    h2d_avg = np.array(h2d_avg)
    d2h_avg = np.array(d2h_avg)
    kernel_avg = np.array(kernel_avg)

    
    graphoutputdir = datafilename.split('/')[1]

    # Plot configs
    slurmid = datafilename.split('/')[2].split('_')[0]
    bandwidths = [
        ('H2DBW', h2d_avg, f'{graphoutputdir}/{slurmid}_h2dbw.png'),
        ('D2HBW', d2h_avg, f'{graphoutputdir}/{slurmid}_d2hbw.png'),
        ('KernelBW', kernel_avg, f'{graphoutputdir}/{slurmid}_kernelbw.png')
    ]

    for name, values, filename in bandwidths:
        plt.figure(figsize=(8, 5))
        plt.xscale('log')
        plt.grid(True)
        plt.xlabel('Data Size (GB)')
        plt.ylabel('Bandwidth (GB/s)')
        plt.title(f'{name} vs Data Size ')
        
        plt.plot(sorted_sizes, values)

        plt.savefig(filename)
        plt.close()  


if __name__ == "__main__":
    main()


