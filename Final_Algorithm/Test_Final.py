import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0

class MultiBiasRRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=3.0, goal_sample_rate=5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacle_list
        self.node_list = []
        
        # Statistik untuk Adaptive Logic
        self.recent_collisions = 0
        self.check_window = 20
        self.best_path_cost = float('inf')
        self.path_found = False
        self.best_path_nodes = []

    def planning(self, animation=True):
        self.node_list = [self.start]
        
        for i in range(1000): # Maksimal 1000 iterasi
            # 1. ADAPTIVE LOGIC: Tentukan Bias Mode
            mode = self.determine_mode(i)
            
            # 2. SAMPLING berdasarkan Mode
            rnd_node = self.sample_based_on_mode(mode)

            # 3. Standard RRT Logic (Nearest & Steer)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. Collision Check & Update Statistics
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                self.recent_collisions = max(0, self.recent_collisions - 1) # Reduce collision counter on success
                
                # Cek apakah sampai Goal
                if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dis:
                    final_node = self.steer(new_node, self.goal, self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list):
                        self.path_found = True
                        self.update_best_path(final_node)
                        print(f"Iter: {i} | Jalur Ditemukan! Cost: {final_node.cost:.2f}")
                        if animation: self.draw_graph(rnd_node)
            else:
                self.recent_collisions += 1 # Tambah counter jika tabrakan

            if animation and i % 20 == 0:
                self.draw_graph(rnd_node)

        return self.best_path_nodes

    def determine_mode(self, iter_idx):
        """Logic 'Supervisor' untuk memilih strategi bias"""
        collision_rate = self.recent_collisions / self.check_window
        
        if self.path_found:
            return "OPTIMIZATION"  # Aktifkan Informed / Path Bias
        elif collision_rate > 0.6: 
            return "CLUTTERED"     # Aktifkan Narrow Passage / Obstacle Bias
        else:
            return "EXPLORATION"   # Aktifkan Goal Bias / Random

    def sample_based_on_mode(self, mode):
        """Memilih algoritma sampling berdasarkan mode"""
        p = random.random()

        if mode == "OPTIMIZATION":
            # 60% Informed, 20% Path Bias, 20% Random
            if p < 0.6: return self.sample_informed()
            elif p < 0.8: return self.sample_path_bias()
            else: return self.sample_uniform()

        elif mode == "CLUTTERED":
            # 50% Narrow Passage, 30% Obstacle Bias, 20% Random
            if p < 0.5: return self.sample_narrow_passage()
            elif p < 0.8: return self.sample_obstacle_bias()
            else: return self.sample_uniform()

        else: # EXPLORATION
            # 10% Goal Bias, 90% Uniform
            if p < 0.1: return Node(self.goal.x, self.goal.y)
            else: return self.sample_uniform()

    # --- IMPLEMENTASI 5 ALGORITMA BIAS ---

    # 1. Bias by Sampling (Informed - Ellipse)
    def sample_informed(self):
        if self.best_path_cost == float('inf'): return self.sample_uniform()
        
        # Simple Rejection Sampling dalam bounding box elips
        # (Versi lengkap memerlukan rotasi matriks, ini versi simplifikasi)
        while True:
            rnd = self.sample_uniform()
            dist = math.sqrt((self.start.x - rnd.x)**2 + (self.start.y - rnd.y)**2) + \
                   math.sqrt((self.goal.x - rnd.x)**2 + (self.goal.y - rnd.y)**2)
            if dist < self.best_path_cost:
                return rnd
            # Timeout break untuk mencegah infinite loop
            if random.random() < 0.05: return self.sample_uniform()

    # 2. Narrow Passage (Bridge Test)
    def sample_narrow_passage(self):
        # Mencari titik tengah antara dua rintangan
        for _ in range(5): # Coba 5 kali
            p1 = self.sample_uniform_anywhere()
            if not self.check_collision(p1, self.obstacle_list): # p1 harus di dalam obstacle
                continue 
            
            # Ambil p2 random di sekitar p1
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(2.0, 6.0) # Jarak 'bridge'
            p2 = Node(p1.x + math.cos(angle)*dist, p1.y + math.sin(angle)*dist)
            
            if not self.check_collision(p2, self.obstacle_list): # p2 juga harus di obstacle
                mid_x = (p1.x + p2.x) / 2
                mid_y = (p1.y + p2.y) / 2
                mid = Node(mid_x, mid_y)
                if self.check_collision(mid, self.obstacle_list): # Titik tengah harus FREE
                    return mid
        return self.sample_uniform()

    # 3. Obstacle Bias (Boundary Sampling)
    def sample_obstacle_bias(self):
        # Sample titik di obstacle, lalu geser sedikit ke area bebas
        p = self.sample_uniform_anywhere()
        if not self.check_collision(p, self.obstacle_list):
            # Jika jatuh di obstacle, coba cari boundary
            # Simplifikasi: geser acak sampai keluar
            for _ in range(5):
                angle = random.uniform(0, 2*math.pi)
                dist = 1.0
                p_new = Node(p.x + math.cos(angle)*dist, p.y + math.sin(angle)*dist)
                if self.check_collision(p_new, self.obstacle_list):
                    return p_new
        return self.sample_uniform()

    # 4. Path Bias
    def sample_path_bias(self):
        if not self.best_path_nodes: return self.sample_uniform()
        
        # Pilih node acak dari jalur yang sudah ada
        target_node = random.choice(self.best_path_nodes)
        # Tambahkan noise (Gaussian perturbation)
        sigma = 2.0
        new_x = target_node.x + random.gauss(0, sigma)
        new_y = target_node.y + random.gauss(0, sigma)
        return Node(new_x, new_y)

    # 5. Uniform / Goal Bias (Standard)
    def sample_uniform(self):
        return Node(random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand))
    
    def sample_uniform_anywhere(self):
        # Sample tanpa peduli collision (untuk bridge test)
        return Node(random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand))

    # --- HELPER FUNCTIONS ---
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        if extend_length > d:
            extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.path_x.append(new_node.x)
        new_node.path_y.append(new_node.y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.goal.x, y - self.goal.y)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def check_collision(self, node, obstacle_list):
        # Mengembalikan True jika AMAN, False jika TABRAKAN
        if node is None: return False
        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= size**2:
                return False  # collision
        return True  # safe

    def update_best_path(self, goal_node):
        path = []
        node = goal_node
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)
        self.best_path_nodes = path
        self.best_path_cost = goal_node.cost

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        
        # Gambar Best Path
        if self.path_found:
             for node in self.best_path_nodes:
                if node.parent:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-r", linewidth=2.0)
                    
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def plot_circle(self, x, y, size, color="-b"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

def main():
    print("Mulai Multi-Bias RRT Path Planning...")
    
    # Setup Obstacles (Circle: x, y, radius)
    # Kita buat "Narrow Passage" di tengah (antara (5,5) dan (7,7))
    obstacle_list = [
        (5, 5, 2),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
    ]

    # Start and Goal
    rrt = MultiBiasRRT(
        start=[0, 0],
        goal=[10, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list
    )
    
    path = rrt.planning(animation=True)

    if path:
        print("Path found!!")
        # Keep window open
        plt.show()
    else:
        print("Cannot find path")

if __name__ == '__main__':
    main()