import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# --- KONFIGURASI UMUM ---
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = [x]
        self.path_y = [y]
        self.parent = None
        self.cost = 0.0

# ==========================================
# 1. KELAS ALGORITMA SKRIPSI (MULTI-BIAS)
# ==========================================
class MultiBiasRRT:
    def __init__(self, start, goal, obstacle_list, rand_area, max_iter=2000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand, self.max_rand = rand_area
        self.obstacle_list = obstacle_list
        self.max_iter = max_iter
        self.node_list = []
        self.expand_dis = 1.0 # Step size kecil biar kelihatan bedanya
        self.path_resolution = 0.5

    def planning(self):
        self.node_list = [self.start]
        start_time = time.time()
        
        for i in range(self.max_iter):
            # --- LOGIKA MULTI-BIAS ---
            p = random.random()
            if p < 0.10:   # 10% Goal Bias
                rnd_node = Node(self.goal.x, self.goal.y)
            elif p < 0.60: # 50% GAUSSIAN BIAS (Focus Narrow Passage)
                rnd_node = self.get_gaussian_bridge_node()
            else:          # 40% Uniform
                rnd_node = Node(random.uniform(self.min_rand, self.max_rand),
                                random.uniform(self.min_rand, self.max_rand))
            
            # Extend
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            new_node = self.steer(self.node_list[nearest_ind], rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                
                # Check Goal
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.expand_dis:
                    final_path = self.generate_final_course(len(self.node_list) - 1)
                    return final_path, self.node_list, time.time() - start_time, i

        return None, self.node_list, time.time() - start_time, self.max_iter

    def get_gaussian_bridge_node(self):
        # Trik Gaussian untuk mencari celah
        sigma = 1.0
        for _ in range(5):
            x1 = random.uniform(self.min_rand, self.max_rand)
            y1 = random.uniform(self.min_rand, self.max_rand)
            safe1 = self.check_collision_point(x1, y1)
            
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            safe2 = self.check_collision_point(x2, y2)

            # Jika satu aman, satu nabrak = TEPIAN HAMBATAN (Potensi Celah)
            if safe1 != safe2:
                return Node((x1+x2)/2, (y1+y2)/2)
        return Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))

    # --- FUNGSI HELPER ---
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.parent = from_node
        return new_node

    def check_collision(self, node, obstacle_list):
        # Simple Point check for speed
        return self.check_collision_point(node.x, node.y)

    def check_collision_point(self, x, y):
        # Batas Map
        if x < self.min_rand or x > self.max_rand or y < self.min_rand or y > self.max_rand:
            return False
        # Obstacle
        for (ox, oy, w, h) in self.obstacle_list:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return False # Nabrak
        return True # Aman

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

# ==========================================
# 2. KELAS RRT POLOS (UNTUK COMPARE)
# ==========================================
class StandardRRT(MultiBiasRRT):
    def planning(self):
        self.node_list = [self.start]
        start_time = time.time()
        
        for i in range(self.max_iter):
            # --- LOGIKA POLOS (RANDOM MURNI) ---
            # Tidak ada Gaussian, Tidak ada strategi narrow
            if random.randint(0, 100) > 5:
                rnd_node = Node(random.uniform(self.min_rand, self.max_rand),
                                random.uniform(self.min_rand, self.max_rand))
            else:
                rnd_node = Node(self.goal.x, self.goal.y)
            
            # Extend (Sama persis logikanya)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            new_node = self.steer(self.node_list[nearest_ind], rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.expand_dis:
                    final_path = self.generate_final_course(len(self.node_list) - 1)
                    return final_path, self.node_list, time.time() - start_time, i
                    
        return None, self.node_list, time.time() - start_time, self.max_iter

# ==========================================
# 3. MAIN PROGRAM (VISUALISASI PERBANDINGAN)
# ==========================================
def main():
    # --- BUAT MAP COMPLEX (ZIG-ZAG MAZE) ---
    # Kotak (x, y, w, h)
    obstacle_list = [
        # Dinding Luar
        (-5, -5, 60, 5),    # Bawah
        (-5, 50, 60, 5),    # Atas
        (-5, 0, 5, 50),     # Kiri
        (50, 0, 5, 50),     # Kanan
        
        # Dinding Penyekat Horizontal (Membentuk Maze)
        (0, 12, 40, 3),     # Sekat 1 (Celah di Kanan)
        (10, 25, 40, 3),    # Sekat 2 (Celah di Kiri)
        (0, 38, 40, 3),     # Sekat 3 (Celah di Kanan)
    ]
    
    start_pos = [5, 5]    # Pojok Bawah
    goal_pos = [5, 45]    # Pojok Atas (Harus lewat 3 celah sempit)
    rand_area = [-5, 55]

    print("="*40)
    print("BENCHMARK: MULTI-BIAS vs STANDARD RRT")
    print("Map: Zig-Zag Narrow Passage")
    print("="*40)

    # --- RUN 1: STANDARD RRT ---
    print("\nRunning Standard RRT (Polos)...")
    rrt_std = StandardRRT(start_pos, goal_pos, obstacle_list, rand_area, max_iter=3000)
    path_std, nodes_std, time_std, iter_std = rrt_std.planning()
    print(f"Hasil Standard: {iter_std} iterasi | {time_std:.2f} detik | {'SUKSES' if path_std else 'GAGAL'}")

    # --- RUN 2: MULTI-BIAS RRT ---
    print("\nRunning Multi-Bias RRT (Gaussian)...")
    rrt_bias = MultiBiasRRT(start_pos, goal_pos, obstacle_list, rand_area, max_iter=3000)
    path_bias, nodes_bias, time_bias, iter_bias = rrt_bias.planning()
    print(f"Hasil Multi-Bias: {iter_bias} iterasi | {time_bias:.2f} detik | {'SUKSES' if path_bias else 'GAGAL'}")

    # --- PLOTTING SIDE-BY-SIDE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Fungsi Gambar Map
    def draw_map(ax, obstacles, nodes, path, title):
        ax.set_title(title)
        ax.set_xlim(-5, 55)
        ax.set_ylim(-5, 55)
        ax.set_aspect('equal')
        
        # Gambar Obstacle
        for (ox, oy, w, h) in obstacles:
            rect = plt.Rectangle((ox, oy), w, h, color='black')
            ax.add_patch(rect)
            
        # Gambar Start/Goal
        ax.plot(start_pos[0], start_pos[1], "xr", markersize=10)
        ax.plot(goal_pos[0], goal_pos[1], "xb", markersize=10)
        
        # Gambar Nodes (Tree)
        # Kita plot titik saja biar ringan, garis bikin berat kalau ribuan
        if nodes:
            nx = [n.x for n in nodes]
            ny = [n.y for n in nodes]
            ax.scatter(nx, ny, c='green', s=1, alpha=0.5)
            
        # Gambar Path
        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, "-r", linewidth=2.5)
            ax.text(25, -4, "SUCCESS", color='red', fontweight='bold', ha='center')
        else:
            ax.text(25, -4, "FAILED", color='black', fontweight='bold', ha='center')

    # Visualisasi 1
    draw_map(ax1, obstacle_list, nodes_std, path_std, 
             f"Standard RRT\nNodes: {len(nodes_std)} | Time: {time_std:.2f}s")

    # Visualisasi 2
    draw_map(ax2, obstacle_list, nodes_bias, path_bias, 
             f"Multi-Bias RRT (Skripsi)\nNodes: {len(nodes_bias)} | Time: {time_bias:.2f}s")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()