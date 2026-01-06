# rrt_standard.py
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
        self.path_x = []
        self.path_y = []

class StandardRRT:
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
        print("Mulai Standard RRT (Animasi Aktif)...")
        start_time = time.time()

        if animation:
            plt.figure(figsize=(8, 8))

        for i in range(self.max_iter):
            # 1. Sampling
            if random.randint(0, 100) > 5:
                rnd = Node(random.uniform(self.min_rand, self.max_rand),
                           random.uniform(self.min_rand, self.max_rand))
            else:
                rnd = Node(self.goal.x, self.goal.y)

            # 2. Extend
            nearest_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd, self.expand_dis)

            if self.check_collision(new_node):
                self.node_list.append(new_node)
                
                # Cek Goal
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.expand_dis:
                    print(f"Goal Found! Iter: {i}, Time: {time.time()-start_time:.2f}s")
                    path = self.generate_final_path(len(self.node_list)-1)
                    if animation: self.draw_graph(rnd, path)
                    return path

            # --- BAGIAN ANIMASI ---
            # Update gambar setiap 50 iterasi agar tidak terlalu berat
            if animation and i % 50 == 0:
                self.draw_graph(rnd)
        
        print("Gagal menemukan jalur.")
        if animation: self.draw_graph(None)
        plt.show()
        return None

    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        # Stop animasi jika tombol ESC ditekan
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        # Gambar Obstacle
        for (ox, oy, w, h) in self.obstacle_list:
            rect = plt.Rectangle((ox, oy), w, h, color='black')
            plt.gca().add_patch(rect)
        
        # Gambar Tree
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", linewidth=0.5)
        
        # Gambar Random Point (Target Sampling saat ini)
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k", markersize=3)

        # Gambar Start & Goal
        plt.plot(self.start.x, self.start.y, "xr", markersize=8, label="Start")
        plt.plot(self.goal.x, self.goal.y, "xb", markersize=8, label="Goal")

        # Gambar Path Akhir
        if path:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2.5, label="Final Path")

        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.title(f"Standard RRT - Nodes: {len(self.node_list)}")
        plt.grid(True)
        plt.pause(0.01) # Penting agar window tidak freeze

    # --- Helper Functions (Sama) ---
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_dist_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.path_x.append(new_node.x)
        new_node.path_y.append(new_node.y)
        new_node.parent = from_node
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

    def check_collision(self, node):
        for (ox, oy, w, h) in self.obstacle_list:
            if ox <= node.x <= ox + w and oy <= node.y <= oy + h:
                return False 
        if not (self.min_rand <= node.x <= self.max_rand and self.min_rand <= node.y <= self.max_rand):
            return False
        return True

    def generate_final_path(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

if __name__ == '__main__':
    cfg = MapConfig()
    rrt = StandardRRT(cfg)
    # Panggil dengan animation=True
    path = rrt.planning(animation=True)
    plt.show()