"""
Script for generating the benchmark plots 
"""
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re

def main():
    """
    Latency vs Frame Plot
    """
    input_csv_dir_path = Path("project/results/")
    pattern = re.compile(r"output_([^_]+)_results.csv")
    for file in input_csv_dir_path.iterdir():    
        if str(file.name).endswith(".csv"):
            mode = re.search(pattern, file.name)
            df = pd.read_csv(file)
    
            df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
            df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
            
            df_clean = df.dropna(subset=['frame', 'latency_ms'])
            
            df_clean = df_clean.sort_values(by='frame')
            
            plt.plot(df_clean['frame'], df_clean['latency_ms'], color='blue', linewidth=1.5)
            
            plt.xlabel('Frame Number')
            plt.ylabel('Latency (ms)')
            plt.title(f'Performance Analysis ({mode.group(1)}): Latency vs. Frames')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(f'project/results/{mode.group(1)}_latency_plot.png')
            plt.close()
    return

if __name__ == "__main__":
    main()