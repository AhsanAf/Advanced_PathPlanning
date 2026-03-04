import socket
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import math
import random
import copy 

# ==========================================
# 1. KONFIGURASI KONEKSI & ROBOT
# ==========================================
HOST = '127.0.0.1'
PORT = 10020

# Parameter Robot & RRT
ROBOT_DIAMETER = 0.08         
ROBOT_RADIUS = ROBOT_DIAMETER / 2.0
# Margin tambahan agar robot tidak terlalu mepet tembok
INFLATION_OFFSET = ROBOT_RADIUS + 0.02 

RRT_STEP_LEN = 0.1              
MAX_ITER = 5000                 
COLLISION_CHECK_STEP = 0.02     # Resolusi cek tabrakan (2 cm)

# Global State
CURRENT_MAP_DATA = None 
LAST_CALCULATED_PATH = None 

# ==========================================
# 2. HELPER FUNGSI (COLLISION CHECK)
# ==========================================
def is_collision(x, y, obstacle_list):
    """Cek titik vs Oriented Bounding Box"""
    for (cx, cy, w, h, angle) in obstacle_list:
        dx = x - cx
        dy = y - cy
        rad = math.radians(-angle) 
        lx = dx * math.cos(rad) - dy * math.sin(rad)
        ly = dx * math.sin(rad) + dy * math.cos(rad)
        
        if abs(lx) <= w/2 and abs(ly) <= h/2:
            return True 
    return False

def check_line_collision(p1, p2, obstacle_list):
    """Cek garis dengan sampling resolusi tinggi"""
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = int(dist / COLLISION_CHECK_STEP) + 1
    
    for i in range(steps + 1):
        t = i / steps
        cx = p1[0] + (p2[0] - p1[0]) * t
        cy = p1[1] + (p2[1] - p1[1]) * t
        if is_collision(cx, cy, obstacle_list):
            return True 
    return False

# ==========================================
# 3. KOMUNIKASI TCP/IP
# ==========================================
def send_command_receive_response(command_char):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print(f"-> Mengirim perintah '{command_char}' ke Webots...")
        client_socket.sendall(command_char.encode())

        full_data = ""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk: break
            text_chunk = chunk.decode('utf-8')
            full_data += text_chunk
            if "END" in full_data: break
        
        client_socket.close()
        return full_data
    except Exception as e:
        print(f"ERROR Conn: {e}")
        return None

def send_path_to_webots(path_list):
    if not path_list:
        print("Error: Tidak ada path.")
        return
    msg = f"PATH,{len(path_list)}"
    for p in path_list:
        msg += f",{p[0]:.4f},{p[1]:.4f}"
    
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print(f"-> Mengirim Path ({len(path_list)} steps)...")
        client_socket.sendall(msg.encode())
        client_socket.close()
        print("-> Sukses!")
    except Exception as e:
        print(f"ERROR Kirim Path: {e}")

def send_trigger(char_code):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        client_socket.sendall(char_code.encode())
        client_socket.close()
    except Exception as e:
        print(f"ERROR: {e}")

# ==========================================
# 4. ALGORITMA RRT & SMOOTHING
# ==========================================
class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x; self.y = y; self.parent = None

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.1):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        for i in range(MAX_ITER):
            rnd = self.get_random_node()
            dlist = [(node.x - rnd.x)**2 + (node.y - rnd.y)**2 for node in self.node_list]
            nearest_node = self.node_list[dlist.index(min(dlist))]

            theta = math.atan2(rnd.y - nearest_node.y, rnd.x - nearest_node.x)
            new_node = self.Node(nearest_node.x + self.expand_dis * math.cos(theta),
                                 nearest_node.y + self.expand_dis * math.sin(theta))
            new_node.parent = nearest_node

            if not check_line_collision([nearest_node.x, nearest_node.y], 
                                        [new_node.x, new_node.y], 
                                        self.obstacle_list):
                self.node_list.append(new_node)
                if math.hypot(new_node.x - self.end.x, new_node.y - self.end.y) <= self.expand_dis:
                    if not check_line_collision([new_node.x, new_node.y], 
                                                [self.end.x, self.end.y], 
                                                self.obstacle_list):
                        final_node = self.Node(self.end.x, self.end.y)
                        final_node.parent = new_node
                        return self.generate_path(final_node)
        return None

    def get_random_node(self):
        if random.randint(0, 100) > 10: 
            return self.Node(random.uniform(self.min_rand[0], self.min_rand[1]),
                             random.uniform(self.max_rand[0], self.max_rand[1]))
        return self.Node(self.end.x, self.end.y)

    def generate_path(self, last_node):
        path = [[last_node.x, last_node.y]]
        while last_node.parent:
            last_node = last_node.parent
            path.append([last_node.x, last_node.y])
        return path[::-1]

def path_smoothing(path, max_iter, obstacle_list):
    if not path or len(path) < 3: return path
    new_path = copy.deepcopy(path)
    
    for _ in range(max_iter):
        if len(new_path) < 3: break
        idx1 = random.randint(0, len(new_path) - 2)
        idx2 = random.randint(idx1 + 1, len(new_path) - 1)
        if idx2 - idx1 <= 1: continue

        p1 = new_path[idx1]; p2 = new_path[idx2]
        if not check_line_collision(p1, p2, obstacle_list):
            temp_path = []
            temp_path.extend(new_path[:idx1+1])
            temp_path.extend(new_path[idx2:])
            new_path = temp_path
    return new_path

# ==========================================
# 5. LOGIKA UTAMA (VISUALISASI UPDATED)
# ==========================================
def process_data(do_planning=False):
    global LAST_CALCULATED_PATH
    
    if not CURRENT_MAP_DATA:
        print("Belum ada data peta.")
        return

    lines = CURRENT_MAP_DATA.strip().split('\n')
    start, goal = None, None
    obs_list = []       
    raw_obs = []        
    arena_w, arena_h = 2.0, 2.0

    for line in lines:
        p = line.split(',')
        if p[0] == "ARENA": 
            arena_w, arena_h = float(p[1]), float(p[2])
        elif p[0] == "ROBOT": 
            start = [float(p[1]), float(p[2])]
        elif p[0] == "TARGET": 
            goal = [float(p[1]), float(p[2])]
        elif p[0] in ["OBS", "WALL"]:
            ox, oy, sx, sy, rot = float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
            raw_obs.append((p[0], ox, oy, sx, sy, rot))
            obs_list.append((ox, oy, sx + INFLATION_OFFSET, sy + INFLATION_OFFSET, rot))

    path, smooth_path = None, None
    
    if do_planning and start and goal:
        print("-> Menghitung RRT...")
        rand_area_x = [-arena_w/2, arena_w/2]
        rand_area_y = [-arena_h/2, arena_h/2]
        rrt = RRT(start, goal, obs_list, [rand_area_x, rand_area_y], RRT_STEP_LEN)
        path = rrt.planning()
        if path:
            print(f"-> RRT Selesai. Smoothing...")
            smooth_path = path_smoothing(path, 150, obs_list)
            LAST_CALCULATED_PATH = smooth_path 
        else:
            print("-> Gagal mencari jalan.")

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    mode_text = "Planning Mode" if do_planning else "Mapping Mode"
    ax.set_title(f"Control Center: {mode_text}")
    
    # 1. Gambar Batas Arena
    ax.add_patch(patches.Rectangle((-arena_w/2, -arena_h/2), arena_w, arena_h, fill=False))
    
    # 2. Gambar Obstacle
    # Agar legenda rapi, kita gunakan flag
    label_obs_added = False
    label_inf_added = False

    for (type_o, ox, oy, sx, sy, rot) in raw_obs:
        # A. GAMBAR INFLATED ZONE (Abu-abu) - SELALU DIGAMBAR
        # Ini adalah area "terlarang" bagi titik pusat robot
        inf_sx = sx + INFLATION_OFFSET
        inf_sy = sy + INFLATION_OFFSET
        
        lbl_inf = "Safety Margin (Radius Robot)" if not label_inf_added else None
        
        inf_rect = patches.Rectangle((-inf_sx/2, -inf_sy/2), 
                                     inf_sx, inf_sy, 
                                     facecolor='lightgray', # Warna Abu-abu
                                     edgecolor='gray',
                                     linestyle='--',
                                     alpha=0.6,
                                     label=lbl_inf)
        t_inf = transforms.Affine2D().rotate_deg(rot).translate(ox, oy) + ax.transData
        inf_rect.set_transform(t_inf)
        ax.add_patch(inf_rect)
        if not label_inf_added: label_inf_added = True

        # B. GAMBAR REAL OBSTACLE (Hitam)
        # Digambar setelah Inflated agar menumpuk di atasnya
        lbl_obs = "Real Obstacle" if not label_obs_added else None
        
        rect = patches.Rectangle((-sx/2, -sy/2), sx, sy, 
                                 facecolor='black', # Warna Hitam
                                 edgecolor='black',
                                 label=lbl_obs)
        t = transforms.Affine2D().rotate_deg(rot).translate(ox, oy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        if not label_obs_added: label_obs_added = True

    if start: ax.plot(start[0], start[1], 'bo', markersize=8, label='Robot Start')
    if goal: ax.plot(goal[0], goal[1], 'rx', markersize=10, label='Target')
    
    if path:
        px, py = zip(*path)
        ax.plot(px, py, 'r--', alpha=0.4, linewidth=1, label='RRT Raw')
    if smooth_path:
        sx, sy = zip(*smooth_path)
        ax.plot(sx, sy, 'c-', linewidth=2.5, label='Optimized Path')

    # Taruh legenda di luar agar tidak menutupi peta jika penuh
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.show()

# ==========================================
# 6. MAIN MENU
# ==========================================
if __name__ == "__main__":
    print("\n=== ROBOT CONTROL CENTER (VISUAL FIXED) ===")
    while True:
        print("\n[M] Mapping (Cek Ukuran Obstacle)")
        print("[P] Path Planning")
        print("[E] Eksekusi Path")
        print("[A] Autonomous")
        print("[Q] Quit")
        
        cmd = input("Pilih: ").strip().upper()

        if cmd == 'M':
            data = send_command_receive_response('M')
            if data:
                CURRENT_MAP_DATA = data
                # Saat Mapping, area abu-abu akan langsung terlihat
                process_data(do_planning=False)
        
        elif cmd == 'P':
            if CURRENT_MAP_DATA:
                process_data(do_planning=True)
            else:
                print("! Data Map kosong. Tekan 'M' dulu.")
        
        elif cmd == 'E':
            if LAST_CALCULATED_PATH:
                send_path_to_webots(LAST_CALCULATED_PATH)
            else:
                print("! Belum ada path.")
        elif cmd == 'A': send_trigger('A')
        elif cmd == 'Q': break