import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the log file containing your output
with open(sys.argv[1]) as f:
    content = f.read()

# Regex pattern to capture the relevant data
pattern = re.compile(
    r'D=(\d+),R=(\d+),B=(\d+),T=(\d+),S=(\d+)\n'
    r'V1=([\d.]+),V1_h2d=([\d.]+),V1_kernel=([\d.]+),V1_d2h=([\d.]+)\n'
    r'V2=([\d.]+),V2_h2d=([\d.]+),V2_kernel=([\d.]+),V2_d2h=([\d.]+)',
    re.MULTILINE
)

# Extract and store the data
rows = []
for match in pattern.findall(content):
    row = {
        "D": int(match[0]),
        "V1": float(match[5]),  # Elapsed time for V1
        "V1_h2d": float(match[6]),
        "V1_kernel": float(match[7]),
        "V1_d2h": float(match[8]),
        "V2": float(match[9]),  # Elapsed time for V2
        "V2_h2d": float(match[10]),
        "V2_kernel": float(match[11]),
        "V2_d2h": float(match[12]),
    }
    rows.append(row)

# Create DataFrame from the extracted data
df = pd.DataFrame(rows)

# Print column names to verify they exist
#print("DataFrame columns:", df.columns)

# If no columns are missing, proceed with plotting
if 'V1' in df.columns and 'V2' in df.columns:
    # Plot Elapsed Time vs Data Size
    plt.figure(figsize=(10, 6))
    plt.plot(df['D'], df['D'] / df['V1'], label='V1 Elapsed Time', color='blue')
    plt.plot(df['D'], df['D'] / df['V2'], label='V2 Elapsed Time', color='green')
    plt.xlabel('Data Size (D)')
    plt.ylabel('Elapsed Time (s)')
    plt.title('Elapsed Time vs Data Size')
    plt.legend()
    plt.xscale('log')  # Set x-axis to log scale
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elapsed_time_vs_data_size.png")

    # Plot H2D bandwidth vs Data Size
    plt.figure(figsize=(10, 6))
    plt.plot(df['D'], df['V1_h2d'], label='V1 H2D Bandwidth', color='blue')
    plt.plot(df['D'], df['V2_h2d'], label='V2 H2D Bandwidth', color='green')
    plt.xlabel('Data Size (D)')
    plt.ylabel('H2D Bandwidth (GB/s)')
    plt.title('H2D Bandwidth vs Data Size')
    plt.legend()
    plt.xscale('log')  # Set x-axis to log scale
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("h2d_bandwidth_vs_data_size.png")

    # Plot Kernel bandwidth vs Data Size
    plt.figure(figsize=(10, 6))
    plt.plot(df['D'], df['V1_kernel'], label='V1 Kernel Bandwidth', color='blue')
    plt.plot(df['D'], df['V2_kernel'], label='V2 Kernel Bandwidth', color='green')
    plt.xlabel('Data Size (D)')
    plt.ylabel('Kernel Bandwidth (GB/s)')
    plt.title('Kernel Bandwidth vs Data Size')
    plt.legend()
    plt.xscale('log')  # Set x-axis to log scale
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kernel_bandwidth_vs_data_size.png")

    # Plot D2H bandwidth vs Data Size
    plt.figure(figsize=(10, 6))
    plt.plot(df['D'], df['V1_d2h'], label='V1 D2H Bandwidth', color='blue')
    plt.plot(df['D'], df['V2_d2h'], label='V2 D2H Bandwidth', color='green')
    plt.xlabel('Data Size (D)')
    plt.ylabel('D2H Bandwidth (GB/s)')
    plt.title('D2H Bandwidth vs Data Size')
    plt.legend()
    plt.xscale('log')  # Set x-axis to log scale
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("d2h_bandwidth_vs_data_size.png")

print("All plots saved as PNG files.")
