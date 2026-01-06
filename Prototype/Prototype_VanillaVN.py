import math
import random
import matplotlib.pyplot as plt
import numpy as np

# --- CLASS NODE SAMA ---
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = [x]
        self.path_y = [y]
        self.parent = None

# --- ALGORITMA RRT POLOS (STANDARD) ---
class StandardRRT:
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=2.0, path_resolution=0.5, 
                 goal_sample_rate=5, max_iter=1000):
        
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.max_iter = max_iter

    def planning(self, animation=True):
        self.node_list = [self.start]
        
        if animation:
            print("Memulai STANDARD RRT (Polos)...")
            print("Note: Tanpa Gaussian Bias & Tanpa Rewiring")
            self.draw_graph()
            plt.pause(1.0)

        for i in range(self.max_iter):
            # 1. SAMPLING BIASA (Uniform + Sedikit Goal Bias)
            # Tidak ada strategi 'Narrow Passage' atau 'Gaussian' disini
            rnd_node = self.get_random_node()

            # 2. CARI NODE TERDEKAT
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. STEER (Gerak ke arah random)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. CEK COLLISION
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                
                # Cek apakah sampai ke Goal
                dx = new_node.x - self.goal.x
                dy = new_node.y - self.goal.y
                d = math.hypot(dx, dy)

                if d <= self.expand_dis:
                    print(f"Goal Ditemukan pada iterasi ke-{i}!")
                    final_node = self.steer(new_node, self.goal, self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list):
                        return self.generate_final_course(len(self.node_list) - 1)

            # Update animasi
            if animation and i % 20 == 0:
                self.draw_graph(rnd_node)

        print("Gagal menemukan jalur (Max Iteration Reached).")
        return None

    # --- FUNGSI SAMPLING POLOS ---
    def get_random_node(self):
        # Hanya mengandalkan keberuntungan (Random Uniform)
        # 5% peluang nembak ke goal, 95% nembak sembarang tempat
        if random.randint(0, 100) > self.goal_sample_rate:
            return Node(random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand))
        else:
            return Node(self.goal.x, self.goal.y)

    # --- FUNGSI PENDUKUNG (SAMA PERSIS) ---
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        
        if extend_length > d: extend_length = d
        n_expand = math.floor(extend_length / self.path_resolution)
        
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y
            
        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

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
        if node is None: return False
        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= (size + 0.1)**2: 
                return False 
        return True 

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None: plt.plot(rnd.x, rnd.y, "^k")
        
        # Gambar Tree
        for node in self.node_list:
            if node.parent: plt.plot(node.path_x, node.path_y, "-g", linewidth=0.5)
            
        # Gambar Obstacle
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)
            
        plt.plot(self.start.x, self.start.y, "xr", markersize=10, label="Start")
        plt.plot(self.goal.x, self.goal.y, "xb", markersize=10, label="Goal")
        
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.title(f"STANDARD RRT (Polos) - Nodes: {len(self.node_list)}")
        plt.pause(0.01)

    def plot_circle(self, x, y, size, color="-b"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

def main():
    # --- SETUP MAP (SAMA PERSIS) ---
    obstacle_list = []
    # Tembok Bawah
    for y in np.arange(-5, 4.0, 1.5): obstacle_list.append((5, y, 1.5))
    # Tembok Atas
    for y in np.arange(6.5, 16, 1.5): obstacle_list.append((5, y, 1.5))
    # Bibir Celah
    obstacle_list.append((5, 3.8, 0.8))
    obstacle_list.append((5, 6.2, 0.8))
    # Clutter
    obstacle_list.append((2, 2, 1.0))
    obstacle_list.append((2, 8, 1.0))
    obstacle_list.append((8, 2, 1.0))
    obstacle_list.append((8, 8, 1.0))

    # --- JALANKAN RRT BIASA ---
    rrt = StandardRRT(
        start=[0, 0],
        goal=[10, 10],
        rand_area=[-2, 12],
        obstacle_list=obstacle_list,
        max_iter=1000 
    )
    
    path = rrt.planning(animation=True)

    if path:
        print("Jalur Ditemukan (Standard RRT)")
        rrt.draw_graph()
        path_x = [x[0] for x in path]
        path_y = [x[1] for x in path]
        # Gambar jalur dengan warna BIRU (beda warna biar kerasa bedanya)
        plt.plot(path_x, path_y, "-b", linewidth=3.5, label="Standard Path")
        plt.legend()
        plt.show()
    else:
        print("GAGAL: Standard RRT kesulitan menembus jalan sempit.")
        plt.show()

if __name__ == '__main__':
    main()