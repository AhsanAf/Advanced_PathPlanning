import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        self.node_list = [self.start]
        
        for i in range(1000): # Maksimal iterasi
            # 1. SIMPLE SAMPLING (Standard)
            rnd_node = self.get_random_node()
            
            # 2. FIND NEAREST
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. STEER (Expand tree)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. COLLISION CHECK
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node) # Tambahkan ke pohon

                # Cek apakah dekat dengan Goal
                dx = new_node.x - self.goal.x
                dy = new_node.y - self.goal.y
                d = math.hypot(dx, dy)

                if d <= self.expand_dis:
                    final_node = self.steer(new_node, self.goal, self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list):
                        print(f"Iter: {i} | Goal Found! (Standard RRT)")
                        return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

        return None  # Failed

    def get_random_node(self):
        # Logic Standard: 95% Random, 5% Goal Bias
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # Goal Bias
            rnd = Node(self.goal.x, self.goal.y)
        return rnd

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # Interpolasi langkah kecil agar collision check akurat
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
            path.append([node.x, node.parent.x]) # Simpan untuk plotting saja
            path.append([node.y, node.parent.y])
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
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_collision(self, node, obstacle_list):
        if node is None: return False
        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= size**2:
                return False  # Collision detected
        return True  # Safe

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        
        # Gambar Tree
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # Gambar Obstacles
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        
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
    print("Mulai Standard RRT Path Planning...")
    
    # OBSTACLE YANG SAMA DENGAN KODE SEBELUMNYA
    obstacle_list = [
        (5, 5, 2),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
    ]

    # Parameter setup
    rrt = RRT(
        start=[0, 0],
        goal=[10, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list
    )
    path = rrt.planning(animation=True)

    if path:
        print("Path found!!")
        # Draw final path
        node = rrt.node_list[-1]
        while node.parent is not None:
             plt.plot(node.path_x, node.path_y, "-r", linewidth=2.5) # Jalur Merah
             node = node.parent
        plt.show()
    else:
        print("Failed to find path")

if __name__ == '__main__':
    main()