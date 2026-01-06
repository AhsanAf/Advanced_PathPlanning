# rrt_multibias.py
import math
import random
import time
import matplotlib.pyplot as plt
from map_config import MapConfig

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class MultiBiasRRT:
    def __init__(self, map_cfg, max_iter=5000):
        self.start = Node(map_cfg.start_pos[0], map_cfg.start_pos[1])
        self.goal = Node(map_cfg.goal_pos[0], map_cfg.goal_pos[1])
        self.min_rand = map_cfg.min_rand
        self.max_rand = map_cfg.max_rand
        self.obstacle_list = map_cfg.obstacle_list
        self.max_iter = max_iter
        self.node_list = []
        self.expand_dis = 2.0 

    def planning(self, animation=True):
        self.node_list = [self.start]
        print("Mulai Multi-Bias RRT (Animasi Aktif)...")
        start_time = time.time()

        if animation:
            plt.figure(figsize=(8, 8))

        # ============================================================
        # LOOP UTAMA RRT
        # Di sini proses pencarian jalur terjadi berulang-ulang
        # ============================================================
        for i in range(self.max_iter):
            
            # --------------------------------------------------------
            # [BAGIAN 1] PEMILIHAN STRATEGI (MULTI-BIAS SELECTION)
            # Dosen: "Dimana letak pembagian persen algoritmanya?"
            # Jawab: "Di blok if-elif-else ini Pak."
            # --------------------------------------------------------
            p = random.random() # Mengocok angka acak 0.0 s/d 1.0
            
            # --- ALGORITMA 1: GOAL BIAS (10%) ---
            # Fungsinya: Memaksa pohon tumbuh lurus ke arah tujuan (Greedy).
            if p < 0.1:
                rnd = Node(self.goal.x, self.goal.y)

            # --- ALGORITMA 2: GAUSSIAN BIAS / NARROW PASSAGE (50%) ---
            # Fungsinya: Mencari celah sempit dengan mendeteksi tepi halangan.
            # (0.1 + 0.5 = 0.6)
            elif p < 0.6:
                rnd = self.get_gaussian_sample() # <--- Memanggil fungsi khusus di bawah

            # --- ALGORITMA 3: UNIFORM SAMPLING (40%) ---
            # Fungsinya: Menjaga eksplorasi global agar tidak macet (Probabilistic Completeness).
            # (Sisa persentase otomatis masuk sini)
            else:
                rnd = Node(random.uniform(self.min_rand, self.max_rand),
                           random.uniform(self.min_rand, self.max_rand))

            # --------------------------------------------------------
            # [BAGIAN 2] EKSPANSI POHON (CORE RRT)
            # Bagian ini standar: Cari Tetangga -> Tarik Garis -> Cek Tabrakan
            # --------------------------------------------------------
            
            # 1. Nearest Neighbor (Cari bapak terdekat)
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # 2. Steering (Batasi langkah robot sejauh expand_dis)
            new_node = self.steer(nearest_node, rnd, self.expand_dis)

            # 3. Collision Check (Validasi: Apakah nabrak tembok?)
            if self.check_collision(new_node):
                self.node_list.append(new_node) # Jika aman, masukkan ke pohon
                
                # Cek apakah sudah sampai Goal?
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.expand_dis:
                    print(f"Goal Found! Iter: {i}, Time: {time.time()-start_time:.2f}s")
                    path = self.generate_final_path(len(self.node_list)-1)
                    if animation: self.draw_graph(rnd, path)
                    return path
            
            # --- BAGIAN ANIMASI (VISUALISASI) ---
            if animation and i % 30 == 0:
                self.draw_graph(rnd)
        
        print("Gagal menemukan jalur.")
        if animation: self.draw_graph(None)
        plt.show()
        return None

    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        for (ox, oy, w, h) in self.obstacle_list:
            rect = plt.Rectangle((ox, oy), w, h, color='black')
            plt.gca().add_patch(rect)
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", linewidth=0.5)
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k", markersize=3)
        plt.plot(self.start.x, self.start.y, "xr", markersize=8, label="Start")
        plt.plot(self.goal.x, self.goal.y, "xb", markersize=8, label="Goal")
        if path:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2.5, label="Final Path")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.title(f"Multi-Bias RRT (Gaussian) - Nodes: {len(self.node_list)}")
        plt.grid(True)
        plt.pause(0.01)

    # --------------------------------------------------------
    # [BAGIAN 3] FUNGSI KHUSUS GAUSSIAN (Bridge Test)
    # Dosen: "Bagaimana cara kerjanya mendeteksi lorong sempit?"
    # Jawab: "Dengan membandingkan dua titik (aman vs nabrak) Pak."
    # --------------------------------------------------------
    def get_gaussian_sample(self):
        sigma = 2.0 # Standar deviasi (Radius sebaran titik kedua)
        
        for _ in range(5): # Coba 5 kali sebelum menyerah
            # 1. Ambil Titik Acak Pertama (Node 1)
            x1 = random.uniform(self.min_rand, self.max_rand)
            y1 = random.uniform(self.min_rand, self.max_rand)
            node1_safe = self.is_point_safe(x1, y1)

            # 2. Ambil Titik Kedua di dekatnya menggunakan distribusi Gaussian (Node 2)
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            node2_safe = self.is_point_safe(x2, y2)

            # 3. DETEKSI TEPI (Boundary Detection Logic)
            # Jika satu titik AMAN dan satu titik NABRAK (XOR),
            # Berarti di antara mereka pasti ada DINDING / BIBIR LORONG.
            if node1_safe != node2_safe:
                # Ambil titik tengahnya (Midpoint) agar pas di pinggir halangan
                return Node((x1+x2)/2, (y1+y2)/2)
        
        # Jika gagal mendapatkan kondisi di atas, kembalikan titik random biasa
        return Node(random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand))

    # --------------------------------------------------------
    # [BAGIAN 4] FUNGSI PENDUKUNG (VALIDASI & MATEMATIKA)
    # --------------------------------------------------------
    def is_point_safe(self, x, y):
        # Cek apakah koordinat (x,y) ada di dalam kotak hitam (Obstacle)
        for (ox, oy, w, h) in self.obstacle_list:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return False # Nabrak
        # Cek apakah keluar dari peta
        if not (self.min_rand <= x <= self.max_rand and self.min_rand <= y <= self.max_rand):
            return False # Keluar Peta
        return True # Aman

    def check_collision(self, node):
        return self.is_point_safe(node.x, node.y)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        # Membatasi langkah agar tidak "teleportasi" terlalu jauh
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_dist_angle(new_node, to_node)
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.parent = from_node # Mencatat orang tua untuk Backtracking jalur
        return new_node

    def calc_dist_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def get_nearest_list_index(self, nodes, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in nodes]
        return dlist.index(min(dlist))

    def generate_final_path(self, goal_ind):
        # Backtracking: Menarik garis merah dari Goal mundur ke Start
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

if __name__ == '__main__':
    cfg = MapConfig()
    rrt = MultiBiasRRT(cfg)
    path = rrt.planning(animation=True)
    plt.show()