import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from map_config import MapConfig
from rrt_standard import StandardRRT
from rrt_multibias import MultiBiasRRT

def calculate_path_length(path):
    if not path: return None
    length = 0
    for i in range(len(path) - 1):
        dx = path[i][0] - path[i+1][0]
        dy = path[i][1] - path[i+1][1]
        length += np.hypot(dx, dy)
    return length

def run_benchmark(num_trials=30, max_iter=1000):
    print(f"=== MEMULAI BENCHMARK ({num_trials} Trials, {max_iter} Iterasi/Trial) ===")
    
    # Simpan Data
    results = {
        'Method': [],
        'Success': [],
        'Cost': [],
        'Time': [],    # Kolom Waktu
        'Nodes': []    # Kolom Jumlah Node
    }

    cfg = MapConfig()

    # --- LOOP PENGUJIAN ---
    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}...", end='\r')
        
        # ==========================
        # 1. TEST STANDARD RRT
        # ==========================
        rrt_std = StandardRRT(cfg, max_iter=max_iter)
        
        # Ukur Waktu Eksternal
        start_time = time.time()
        path_std = rrt_std.planning(animation=False) 
        exec_time = time.time() - start_time
        
        # Ambil Data
        success_std = 1 if path_std is not None else 0
        cost_std = calculate_path_length(path_std) if path_std else None
        node_count_std = len(rrt_std.node_list) # Ambil jumlah node langsung dari objek
        
        # Masukkan ke Tabel
        results['Method'].append('Standard')
        results['Success'].append(success_std)
        results['Cost'].append(cost_std)
        results['Time'].append(exec_time)
        results['Nodes'].append(node_count_std)
        
        # ==========================
        # 2. TEST MULTI-BIAS RRT
        # ==========================
        rrt_bias = MultiBiasRRT(cfg, max_iter=max_iter)
        
        # Ukur Waktu Eksternal
        start_time = time.time()
        path_bias = rrt_bias.planning(animation=False)
        exec_time = time.time() - start_time
        
        # Ambil Data
        success_bias = 1 if path_bias is not None else 0
        cost_bias = calculate_path_length(path_bias) if path_bias else None
        node_count_bias = len(rrt_bias.node_list)
        
        # Masukkan ke Tabel
        results['Method'].append('Multi-Bias')
        results['Success'].append(success_bias)
        results['Cost'].append(cost_bias)
        results['Time'].append(exec_time)
        results['Nodes'].append(node_count_bias)

    print("\nBenchmark Selesai. Mengolah Data...")
    return pd.DataFrame(results)

# --- ANALISIS DATA ---
def analyze_results(df):
    print("\n" + "="*30)
    print("HASIL PERBANDINGAN QUANTITATIVE")
    print("="*30)
    
    # 1. Success Rate
    success_rate = df.groupby('Method')['Success'].mean() * 100
    print("\n[1] SUCCESS RATE (% Berhasil):")
    print(success_rate)
    
    # 2. Path Cost (Hanya hitung rata-rata jika sukses)
    print("\n[2] AVERAGE PATH COST (Makin Kecil Makin Bagus):")
    print(df[df['Cost'].notnull()].groupby('Method')['Cost'].mean())

    # 3. Nodes Count
    print("\n[3] AVERAGE NODES (Makin Sedikit = Memori Hemat):")
    print(df.groupby('Method')['Nodes'].mean())
    
    # 4. Execution Time
    print("\n[4] AVERAGE TIME (Detik):")
    print(df.groupby('Method')['Time'].mean())
    
    # --- VISUALISASI ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # A. Success Rate (Bar Chart)
    methods = success_rate.index
    rates = success_rate.values
    ax[0, 0].bar(methods, rates, color=['red', 'blue'], alpha=0.7)
    ax[0, 0].set_title("Success Rate (%)")
    ax[0, 0].set_ylim(0, 110)
    
    # B. Path Cost (Boxplot) - Filter NaN
    data_cost = []
    labels_cost = []
    for m in df['Method'].unique():
        subset = df[(df['Method'] == m) & (df['Cost'].notnull())]['Cost']
        if len(subset) > 0:
            data_cost.append(subset)
            labels_cost.append(m)
    
    if data_cost:
        ax[0, 1].boxplot(data_cost, labels=labels_cost)
        ax[0, 1].set_title("Path Cost (Distance)")
    
    # C. Node Count (Bar Chart Rata-rata)
    avg_nodes = df.groupby('Method')['Nodes'].mean()
    ax[1, 0].bar(avg_nodes.index, avg_nodes.values, color=['red', 'blue'], alpha=0.7)
    ax[1, 0].set_title("Average Nodes Generated")
    
    # D. Time (Bar Chart Rata-rata)
    avg_time = df.groupby('Method')['Time'].mean()
    ax[1, 1].bar(avg_time.index, avg_time.values, color=['red', 'blue'], alpha=0.7)
    ax[1, 1].set_title("Average Execution Time (s)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Jalankan benchmark
    # Tips: Gunakan max_iter=2000 untuk map yang sulit (shortcut vs detour)
    df_results = run_benchmark(num_trials=20, max_iter=2000)
    analyze_results(df_results)