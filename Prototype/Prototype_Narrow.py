import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = [x] 
        self.path_y = [y]
        self.parent = None

class AnytimeNarrowPassageRRT:
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=2.0, path_resolution=0.5, 
                 goal_sample_rate=5, narrow_passage_ratio=30, max_iter=1000):
        
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.narrow_passage_ratio = narrow_passage_ratio
        self.max_iter = max_iter # Batas iterasi tetap
        
        # Penampung jalur-jalur yang sukses
        self.found_paths = [] 

    def planning(self, animation=True):
        self.node_list = [self.start]
        self.found_paths = [] # Reset
        
        if animation:
            print(f"Mulai Anytime RRT (Max Iter: {self.max_iter})...")
            self.draw_graph()
            plt.pause(1.0)

        for i in range(self.max_iter):
            # 1. SAMPLING STRATEGY
            if random.randint(0, 100) < self.narrow_passage_ratio:
                rnd_node, _ = self.sample_bridge_node()
            else:
                rnd_node = self.get_standard_random_node()

            # 2. STANDARD RRT STEPS
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                
                # 3. CEK GOAL (TAPI JANGAN STOP)
                dx = new_node.x - self.goal.x
                dy = new_node.y - self.goal.y
                d = math.hypot(dx, dy)

                if d <= self.expand_dis:
                    final_node = self.steer(new_node, self.goal, self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list):
                        # Simpan jalur ini, tapi JANGAN return dulu!
                        path = self.generate_final_course(len(self.node_list) - 1)
                        # Hitung cost (panjang jalur)
                        cost = self.calculate_path_cost(path)
                        self.found_paths.append((path, cost))
                        
                        print(f"Iter: {i} | Jalur Alternatif Ditemukan! Cost: {cost:.2f}")

            # Update animasi visual
            if animation and i % 10 == 0:
                self.draw_graph(rnd_node)

        # 4. SELESAI LOOP - PILIH YANG TERBAIK
        print(f"Selesai {self.max_iter} iterasi.")
        
        if not self.found_paths:
            print("Gagal menemukan jalur satupun.")
            return None
        else:
            # Sortir berdasarkan cost terendah
            self.found_paths.sort(key=lambda x: x[1])
            best_path = self.found_paths[0][0]
            best_cost = self.found_paths[0][1]
            print(f"MEMILIH JALUR TERPENDEK DARI {len(self.found_paths)} OPSI.")
            print(f"Best Cost: {best_cost:.2f}")
            return best_path

    def calculate_path_cost(self, path):
        cost = 0.0
        for i in range(len(path) - 1):
            dx = path[i][0] - path[i+1][0]
            dy = path[i][1] - path[i+1][1]
            cost += math.hypot(dx, dy)
        return cost

    # --- FUNGSI-FUNGSI LAIN TETAP SAMA ---
    def sample_bridge_node(self):
        for _ in range(5):
            p1 = Node(random.uniform(self.min_rand, self.max_rand),
                      random.uniform(self.min_rand, self.max_rand))
            p1.path_x = [p1.x]
            p1.path_y = [p1.y]
            if self.check_collision(p1, self.obstacle_list): continue 
            sigma = 1.5
            p2 = Node(p1.x + random.gauss(0, sigma), p1.y + random.gauss(0, sigma))
            p2.path_x = [p2.x]
            p2.path_y = [p2.y]
            if self.check_collision(p2, self.obstacle_list): continue
            mid_node = Node((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            mid_node.path_x = [mid_node.x]
            mid_node.path_y = [mid_node.y]
            if self.check_collision(mid_node, self.obstacle_list): return mid_node, True
        return self.get_standard_random_node(), False

    def get_standard_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            return Node(random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand))
        else:
            return Node(self.goal.x, self.goal.y)

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
            if min(d_list) <= size**2: return False 
        return True 

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None: plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent: plt.plot(node.path_x, node.path_y, "-g")
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        
        # [VISUALISASI TAMBAHAN] 
        # Gambarkan semua jalur yang SUDAH ditemukan sementara dengan warna oranye tipis
        for path_data in self.found_paths:
            p = path_data[0]
            px = [x[0] for x in p]
            py = [x[1] for x in p]
            plt.plot(px, py, "--", color="orange", linewidth=1.0)

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
    # SKENARIO TEMBOK KOMPLEKS
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

    # SETTING ITERASI DISINI
    MAX_ITER = 1000 

    rrt = AnytimeNarrowPassageRRT(
        start=[0, 0],
        goal=[10, 10],
        rand_area=[-2, 12],
        obstacle_list=obstacle_list,
        narrow_passage_ratio=30,
        max_iter=MAX_ITER # Gunakan max iter
    )
    
    best_path = rrt.planning(animation=True)

    if best_path:
        print("MENGGAMBAR JALUR TERBAIK...")
        path_x = [x[0] for x in best_path]
        path_y = [x[1] for x in best_path]
        # Gambar ulang final path dengan warna MERAH TEBAL
        plt.plot(path_x, path_y, "-r", linewidth=3.5, label="Best Path")
        plt.legend()
        plt.show()
    else:
        print("Failed to find path")

if __name__ == '__main__':
    main()