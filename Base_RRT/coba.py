#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid A* + RRT* (Goal-Bias + Narrow Passage + Fixed Step + A*-Guided + A*-Optimization)
Versi TANPA GUI Tkinter + PETA RANDOM 640x640

- Peta kosong 640x640
- Obstacle acak (hitam):
    * Persegi panjang (rectangle)
    * Persegi (square)
    * Bulat (circle)
  dengan ukuran random, dan DIUSAHAKAN TIDAK SALING MENIMPA.
  Overlap kecil masih boleh (supaya bisa terbentuk huruf L, dsb).

- Start: biru (#0000FF)
- Goal : merah (#FF0000)

Program langsung:
1) generate map + start + goal
2) jalankan RRT* hybrid
3) tampilkan animasi growth tree + final path di jendela matplotlib
"""

import math
import random
import time
import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ======= Animasi ======= #
ANIM_DELAY = 0.001  # jeda per segmen (detik); boleh kamu ubah kalau terlalu cepat / lambat

# -------------------- A* Global Planner (grid-based) -------------------- #

def astar_grid(
    occ_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """
    A* di atas occupancy grid:
    - occ_map: HxW, True=obstacle, False=free
    - start, goal: (x,y) pixel coords
    Menghasilkan path list [(x,y), ...] atau None bila tidak ada path.
    """
    h, w = occ_map.shape
    sx, sy = start
    gx, gy = goal

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h

    if not in_bounds(sx, sy) or not in_bounds(gx, gy):
        return None
    if occ_map[sy, sx] or occ_map[gy, gx]:
        return None

    # 8-connected moves
    neighbors = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    ]

    def heuristic(x: int, y: int) -> float:
        return math.hypot(x - gx, y - gy)

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    g_cost: Dict[Tuple[int, int], float] = {}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    start_node = (sx, sy)
    goal_node = (gx, gy)

    g_cost[start_node] = 0.0
    f0 = heuristic(sx, sy)
    heapq.heappush(open_heap, (f0, 0.0, start_node))

    max_expansions = h * w * 5  # batas aman

    expansions = 0
    while open_heap:
        f, g_curr, (x, y) = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            break

        if (x, y) == goal_node:
            # Rekonstruksi path
            path: List[Tuple[int, int]] = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path.append((x, y))
            path.reverse()
            return path

        if g_curr > g_cost.get((x, y), float("inf")):
            continue

        for dx, dy, w_move in neighbors:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if occ_map[ny, nx]:
                continue

            # Cegah "corner cutting" untuk diagonal
            if dx != 0 and dy != 0:
                if occ_map[y, nx] and occ_map[ny, x]:
                    continue

            new_g = g_curr + w_move
            if new_g < g_cost.get((nx, ny), float("inf")):
                g_cost[(nx, ny)] = new_g
                came_from[(nx, ny)] = (x, y)
                f_new = new_g + heuristic(nx, ny)
                heapq.heappush(open_heap, (f_new, new_g, (nx, ny)))

    return None


# -------------------- RRT* core (with Narrow Passage + A* Hybrid) -------------------- #

@dataclass
class Node:
    x: int
    y: int
    parent: Optional[int] = None
    cost: float = 0.0


class RRTStar:
    def __init__(
        self,
        occ_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        step_len: int = 32,
        goal_sample_rate: float = 0.05,    # goal bias kecil (gaya Sakai)
        neighbor_radius: int = 60,         # upper bound radius rewire
        max_iter: int = 4000,

        # Narrow Passage
        narrow_bias_rate: float = 0.12,
        bridge_sigma: float = 10.0,
        bridge_trials: int = 5,

        # "Sakai-like" visual controls
        fixed_step: bool = True,           # segmen selalu sepanjang step_len (jika memungkinkan)
        min_sample_sep: int = 14,          # tolak sampel terlalu dekat node (px)
        gamma_rewire: float = 90.0,        # konstanta radius adaptif (px)

        # Hybrid dengan A*
        astar_guided_rate: float = 0.30    # probabilitas sampling diarahkan sepanjang jalur A*
    ):
        """
        occ_map: HxW boolean array; True=obstacle, False=free
        start, goal: (x,y) pixel coords
        """
        self.occ = occ_map
        self.h, self.w = occ_map.shape
        self.start = Node(*start, parent=None, cost=0.0)
        self.goal = Node(*goal, parent=None, cost=0.0)

        self.step_len = int(step_len)
        self.goal_sample_rate = float(goal_sample_rate)
        self.neighbor_radius = float(neighbor_radius)
        self.max_iter = int(max_iter)

        # Narrow-passage params
        self.narrow_bias_rate = float(np.clip(narrow_bias_rate, 0.0, 0.9))
        self.bridge_sigma = float(max(1.0, bridge_sigma))
        self.bridge_trials = max(1, int(bridge_trials))

        # Visual/structure controls
        self.fixed_step = bool(fixed_step)
        self.min_sample_sep = int(max(0, min_sample_sep))
        self.gamma_rewire = float(max(1.0, gamma_rewire))

        # Visit grid untuk fallback low-density sampling
        self.visit = np.zeros((self.h, self.w), dtype=np.uint16)

        self.nodes: List[Node] = [self.start]

        # Hybrid: Global A* path (untuk guidance)
        self.global_astar_path: Optional[List[Tuple[int, int]]] = astar_grid(
            self.occ,
            (self.start.x, self.start.y),
            (self.goal.x, self.goal.y),
        )
        self.astar_guided_rate = float(np.clip(astar_guided_rate, 0.0, 1.0))
        if self.global_astar_path is None:
            # Tidak ada path global dari A*, jangan pakai guidance A*
            self.astar_guided_rate = 0.0

    # ------------- Helpers ------------- #
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def _is_free(self, x: int, y: int) -> bool:
        return self._in_bounds(x, y) and (not self.occ[y, x])

    def _rand_xy(self) -> Tuple[int, int]:
        return random.randint(0, self.w - 1), random.randint(0, self.h - 1)

    def _clamp(self, x: float, y: float) -> Tuple[int, int]:
        xi = int(min(max(round(x), 0), self.w - 1))
        yi = int(min(max(round(y), 0), self.h - 1))
        return xi, yi

    def _too_close_to_nodes(self, x: int, y: int) -> bool:
        if self.min_sample_sep <= 0:
            return False
        r2 = self.min_sample_sep * self.min_sample_sep
        for n in self.nodes:
            dx, dy = n.x - x, n.y - y
            if dx * dx + dy * dy <= r2:
                return True
        return False

    # -------- Narrow Passage Sampler -------- #
    def _sample_bridge(self) -> Optional[Tuple[int, int]]:
        """
        Bridge test:
        - dua titik berlawanan (Gaussian offset) keduanya di OBSTACLE
        - midpoint BEBAS -> kandidat koridor sempit
        """
        for _ in range(self.bridge_trials):
            cx, cy = self._rand_xy()
            dx = np.random.normal(0, self.bridge_sigma)
            dy = np.random.normal(0, self.bridge_sigma)
            x1, y1 = self._clamp(cx + dx, cy + dy)
            x2, y2 = self._clamp(cx - dx, cy - dy)
            if not (self._in_bounds(x1, y1) and self._in_bounds(x2, y2)):
                continue
            mx, my = self._clamp((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            if self.occ[y1, x1] and self.occ[y2, x2] and self._is_free(mx, my) and not self._too_close_to_nodes(mx, my):
                return (mx, my)
        return None

    def _sample_low_density(self) -> Optional[Tuple[int, int]]:
        """Fallback: pilih titik bebas dengan kunjungan terendah."""
        tries = 30
        best = None
        best_vis = None
        for _ in range(tries):
            x, y = self._rand_xy()
            if not self._is_free(x, y) or self._too_close_to_nodes(x, y):
                continue
            v = self.visit[y, x]
            if (best is None) or (v < best_vis):
                best, best_vis = (x, y), v
        return best

    # -------- A*-Guided Sampler (Hybrid 1) -------- #
    def _sample_on_global_path(self, jitter_sigma: float = 5.0, trials: int = 20) -> Optional[Tuple[int, int]]:
        """
        Sampling di sekitar jalur global A*:
        - pilih satu segmen pada path A*
        - sampling titik di garis itu + jitter kecil
        """
        if self.global_astar_path is None or len(self.global_astar_path) < 2:
            return None

        for _ in range(trials):
            idx = random.randint(0, len(self.global_astar_path) - 2)
            x1, y1 = self.global_astar_path[idx]
            x2, y2 = self.global_astar_path[idx + 1]
            t = random.random()
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # jitter sedikit
            if jitter_sigma > 0:
                x += random.gauss(0, jitter_sigma)
                y += random.gauss(0, jitter_sigma)

            xi, yi = self._clamp(x, y)
            if self._is_free(xi, yi) and not self._too_close_to_nodes(xi, yi):
                return (xi, yi)

        return None

    def _sample(self) -> Tuple[int, int]:
        # 0) A*-guided sampling (global path guidance)
        if self.astar_guided_rate > 0.0 and random.random() < self.astar_guided_rate:
            s_astar = self._sample_on_global_path()
            if s_astar is not None:
                return s_astar

        # 1) goal bias
        if random.random() < self.goal_sample_rate:
            gx, gy = self.goal.x, self.goal.y
            if not self._too_close_to_nodes(gx, gy):
                return gx, gy

        # 2) narrow passage bias
        if random.random() < self.narrow_bias_rate:
            s = self._sample_bridge()
            if s is not None:
                return s
            s2 = self._sample_low_density()
            if s2 is not None:
                return s2

        # 3) uniform with separation
        for _ in range(20):
            x, y = self._rand_xy()
            if self._is_free(x, y) and not self._too_close_to_nodes(x, y):
                return x, y
        # 4) ultimate fallback
        return self._rand_xy()

    def _nearest(self, x: int, y: int) -> int:
        dists = [(i, (n.x - x) ** 2 + (n.y - y) ** 2) for i, n in enumerate(self.nodes)]
        i_min = min(dists, key=lambda t: t[1])[0]
        return i_min

    def _steer(self, from_node: Node, to_point: Tuple[int, int]) -> Tuple[int, int]:
        fx, fy = from_node.x, from_node.y
        tx, ty = to_point
        dx, dy = tx - fx, ty - fy
        dist = math.hypot(dx, dy)
        if dist == 0:
            return fx, fy
        if self.fixed_step and dist > self.step_len:
            scale = self.step_len / dist          # selalu segmen sepanjang step_len
        else:
            scale = min(self.step_len / dist, 1.0)
        nx = int(round(fx + dx * scale))
        ny = int(round(fy + dy * scale))
        return nx, ny

    def _collision_free_segment(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Bresenham-style check antara dua titik (inklusif)."""
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x, y = x1, y1
        while True:
            if not self._is_free(x, y):
                return False
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return True

    def _near_indices(self, x: int, y: int, radius: float) -> List[int]:
        r2 = radius * radius
        return [i for i, n in enumerate(self.nodes) if (n.x - x) ** 2 + (n.y - y) ** 2 <= r2]

    def _cost(self, idx: int) -> float:
        return self.nodes[idx].cost

    # -------- Graph Construction + A* di Graph (Hybrid 2) -------- #
    def _build_graph_edges(self, radius_opt: float) -> List[List[Tuple[int, float]]]:
        """
        Bangun graf dari node-node RRT:
        - edge antara node i dan j jika jarak <= radius_opt dan bebas collision.
        """
        N = len(self.nodes)
        edges: List[List[Tuple[int, float]]] = [[] for _ in range(N)]

        for i, ni in enumerate(self.nodes):
            neighbor_ids = self._near_indices(ni.x, ni.y, radius_opt)
            for j in neighbor_ids:
                if j == i:
                    continue
                nj = self.nodes[j]
                if self._collision_free_segment((ni.x, ni.y), (nj.x, nj.y)):
                    cost = math.hypot(nj.x - ni.x, nj.y - ni.y)
                    edges[i].append((j, cost))
        return edges

    def _astar_on_graph(
        self,
        edges: List[List[Tuple[int, float]]],
        start_idx: int,
        goal_idx: int
    ) -> Optional[List[int]]:
        """
        A* di graf node-node RRT:
        - edges: adjacency list [ [(neighbor_idx, cost), ...], ... ]
        - start_idx, goal_idx: indeks node start & goal
        """
        N = len(edges)
        g_cost = [float("inf")] * N
        g_cost[start_idx] = 0.0
        came_from = [-1] * N

        def heuristic(i: int) -> float:
            ni = self.nodes[i]
            ng = self.nodes[goal_idx]
            return math.hypot(ni.x - ng.x, ni.y - ng.y)

        open_heap: List[Tuple[float, float, int]] = []
        heapq.heappush(open_heap, (heuristic(start_idx), 0.0, start_idx))

        while open_heap:
            f, g_curr, i = heapq.heappop(open_heap)
            if i == goal_idx:
                # Rekonstruksi path index
                path_idx: List[int] = [i]
                while came_from[i] != -1:
                    i = came_from[i]
                    path_idx.append(i)
                path_idx.reverse()
                return path_idx

            if g_curr > g_cost[i]:
                continue

            for j, w_move in edges[i]:
                new_g = g_curr + w_move
                if new_g < g_cost[j]:
                    g_cost[j] = new_g
                    came_from[j] = i
                    f_new = new_g + heuristic(j)
                    heapq.heappush(open_heap, (f_new, new_g, j))

        return None

    def _optimize_path_via_graph(
        self,
        goal_idx: int,
        raw_idx_path: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Hybrid 2: setelah goal ditemukan via RRT* (tree),
        bangun graf lokal dan jalankan A* di graf itu untuk mencari path terbaik.
        Bila gagal, kembali ke raw path.
        """
        try:
            radius_opt = self.neighbor_radius  # bisa disesuaikan
            edges = self._build_graph_edges(radius_opt)
            idx_path_opt = self._astar_on_graph(edges, start_idx=0, goal_idx=goal_idx)
            if idx_path_opt is None:
                idx_final = raw_idx_path
            else:
                idx_final = idx_path_opt

            path_coords: List[Tuple[int, int]] = []
            for idx in idx_final:
                n = self.nodes[idx]
                path_coords.append((n.x, n.y))
            return path_coords

        except Exception:
            path_coords: List[Tuple[int, int]] = []
            for idx in raw_idx_path:
                n = self.nodes[idx]
                path_coords.append((n.x, n.y))
            return path_coords

    def plan(self, callback_draw_segment=None, callback_draw_best=None) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Jalankan RRT* + Hybrid A*; kembalikan (path, iters)
        - path: list [(x,y), ...] bila ditemukan, atau None bila gagal
        - iters: jumlah iterasi yang dijalankan sampai return
        """
        iters = 0
        for iters in range(1, self.max_iter + 1):
            rx, ry = self._sample()
            if 0 <= ry < self.h and 0 <= rx < self.w:
                self.visit[ry, rx] = min(np.iinfo(self.visit.dtype).max, self.visit[ry, rx] + 1)

            nearest_idx = self._nearest(rx, ry)
            new_x, new_y = self._steer(self.nodes[nearest_idx], (rx, ry))

            if not self._is_free(new_x, new_y):
                continue
            if not self._collision_free_segment((self.nodes[nearest_idx].x, self.nodes[nearest_idx].y), (new_x, new_y)):
                continue

            # ---- Adaptive rewire radius (clamped by neighbor_radius) ----
            n = len(self.nodes) + 1
            radius = max(
                10.0,
                min(
                    self.neighbor_radius,
                    self.gamma_rewire * math.sqrt(max(1.0, math.log(n)) / n),
                ),
            )
            neighbor_ids = self._near_indices(new_x, new_y, radius)

            # ---- Choose parent (minimum total cost) ----
            new_parent = nearest_idx
            new_cost_best = self._cost(nearest_idx) + math.hypot(
                new_x - self.nodes[nearest_idx].x,
                new_y - self.nodes[nearest_idx].y,
            )
            for j in neighbor_ids:
                cand = self.nodes[j]
                if self._collision_free_segment((cand.x, cand.y), (new_x, new_y)):
                    cand_cost = self._cost(j) + math.hypot(new_x - cand.x, new_y - new_y)
                    cand_cost = self._cost(j) + math.hypot(new_x - cand.x, new_y - cand.y)
                    if cand_cost < new_cost_best:
                        new_cost_best = cand_cost
                        new_parent = j

            self.nodes.append(Node(new_x, new_y, parent=new_parent, cost=new_cost_best))
            new_idx = len(self.nodes) - 1

            # ---- Rewire ----
            for j in neighbor_ids:
                if j == new_idx:
                    continue
                cand = self.nodes[j]
                new_cost = self._cost(new_idx) + math.hypot(cand.x - new_x, cand.y - new_y)
                if new_cost + 1e-6 < self._cost(j) and self._collision_free_segment((new_x, new_y), (cand.x, cand.y)):
                    self.nodes[j].parent = new_idx
                    self.nodes[j].cost = new_cost

            # ---- Gambar segmen baru ----
            if callback_draw_segment is not None:
                parent = self.nodes[new_parent]
                callback_draw_segment((parent.x, parent.y, new_x, new_y))

            # ---- Cek koneksi ke goal ----
            if self._collision_free_segment((new_x, new_y), (self.goal.x, self.goal.y)):
                # Tambahkan node goal ke tree
                self.nodes.append(
                    Node(
                        self.goal.x,
                        self.goal.y,
                        parent=new_idx,
                        cost=self._cost(new_idx)
                        + math.hypot(self.goal.x - new_x, self.goal.y - new_y),
                    )
                )
                goal_idx = len(self.nodes) - 1

                # Raw path dari parent chain
                raw_idx_path: List[int] = []
                cur = goal_idx
                while cur is not None:
                    raw_idx_path.append(cur)
                    cur = self.nodes[cur].parent
                raw_idx_path.reverse()

                # Hybrid 2: optimalkan path dengan A* di graf node RRT
                path_coords = self._optimize_path_via_graph(goal_idx, raw_idx_path)

                if callback_draw_best is not None:
                    callback_draw_best(path_coords)
                return path_coords, iters

        return None, iters


# -------------------- Utilities: image/occ + random map -------------------- #

def image_to_occ(img: Image.Image, black_thresh: int = 60) -> np.ndarray:
    """Mask obstacle dari piksel dekat-hitam."""
    arr = np.array(img.convert("RGB")).astype(np.int16)
    obs = (arr[:, :, 0] < black_thresh) & (arr[:, :, 1] < black_thresh) & (arr[:, :, 2] < black_thresh)
    return obs


def generate_random_map(
    width: int = 640,
    height: int = 640,
    n_obstacles: int = 18,
    max_overlap_ratio: float = 0.15,
    max_tries_per_shape: int = 80,
) -> Tuple[Image.Image, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Generate peta:
    - peta putih 640x640
    - obstacle hitam (persegi panjang, persegi, bulat) acak
      dengan aturan: tidak saling menimpa besar-besaran.
      Overlap kecil <= max_overlap_ratio * area shape masih diizinkan
      (bisa membentuk huruf L, dsb).
    - start biru, goal merah (posisi random pada free space)
    """
    # Peta dasar putih
    base_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw_base = ImageDraw.Draw(base_img)

    # Occupancy map manual (boolean)
    occ = np.zeros((height, width), dtype=bool)

    # Pastikan minimal 1 dari tiap bentuk
    shape_types = ["rect", "square", "circle"]
    shapes = []

    # Add at least one of each
    for st in shape_types:
        shapes.append(st)

    # Tambah sisanya random
    for _ in range(max(0, n_obstacles - len(shapes))):
        shapes.append(random.choice(shape_types))

    # Fungsi untuk menggambar shape ke mask sementara
    def make_shape_mask(st: str, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        temp = Image.new("L", (width, height), 0)
        dtemp = ImageDraw.Draw(temp)
        x1, y1 = x0 + w, y0 + h
        if st == "circle":
            dtemp.ellipse((x0, y0, x1, y1), fill=255)
        else:
            dtemp.rectangle((x0, y0, x1, y1), fill=255)
        mask = np.array(temp) > 0
        return mask

    # Gambar obstacles dengan cek overlap
    for st in shapes:
        placed = False
        for _ in range(max_tries_per_shape):
            if st == "rect":
                w = random.randint(50, 160)
                h = random.randint(30, 120)
            elif st == "square":
                s = random.randint(40, 120)
                w = h = s
            else:  # circle
                r = random.randint(25, 80)
                w = h = 2 * r

            if w >= width or h >= height:
                continue

            x0 = random.randint(0, width - w)
            y0 = random.randint(0, height - h)

            # Mask kandidat shape
            cand_mask = make_shape_mask(st if st != "circle" else "circle", x0, y0, w, h)
            new_area = cand_mask.sum()
            if new_area == 0:
                continue

            overlap_pixels = np.logical_and(occ, cand_mask).sum()
            max_overlap = int(max_overlap_ratio * new_area)

            # jika overlap kecil -> OK
            if overlap_pixels <= max_overlap:
                # Gambar ke peta nyata
                x1, y1 = x0 + w, y0 + h
                if st == "circle":
                    draw_base.ellipse((x0, y0, x1, y1), fill=(0, 0, 0))
                else:
                    draw_base.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
                occ |= cand_mask
                placed = True
                break

        # kalau gagal ditempatkan setelah banyak tries, kita skip shape ini
        if not placed:
            print(f"Warning: gagal menempatkan obstacle tipe {st}, diskip.")

    # Pilih start & goal pada free space (tidak di obstacle)
    def random_free_point(min_margin: int = 10) -> Tuple[int, int]:
        while True:
            x = random.randint(min_margin, width - 1 - min_margin)
            y = random.randint(min_margin, height - 1 - min_margin)
            if not occ[y, x]:
                return x, y

    start = random_free_point()
    # Pastikan goal cukup jauh dari start
    while True:
        goal = random_free_point()
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) > width * 0.3:
            break

    # Gambar start (biru) dan goal (merah) di atas peta
    vis_img = base_img.copy()
    dvis = ImageDraw.Draw(vis_img)
    sz = 6  # ukuran marker start/goal
    sx, sy = start
    gx, gy = goal
    dvis.rectangle((sx - sz, sy - sz, sx + sz, sy + sz), fill=(0, 0, 255))   # biru
    dvis.rectangle((gx - sz, gy - sz, gx + sz, gy + sz), fill=(255, 0, 0))   # merah

    return vis_img, occ, start, goal


# -------------------- Visual helper: draw line ke numpy canvas -------------------- #

def draw_line_on_canvas(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int]
):
    """
    Gambar garis (Bresenham) ke canvas numpy [H,W,3] dengan warna RGB.
    """
    h, w, _ = canvas.shape

    def clamp_xy(x, y):
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    x1, y1 = clamp_xy(x1, y1)
    x2, y2 = clamp_xy(x2, y2)

    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    x, y = x1, y1

    while True:
        if 0 <= x < w and 0 <= y < h:
            canvas[y, x, :] = color
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


# -------------------- MAIN: random map + animasi RRT -------------------- #

def main():
    random.seed()  # bisa diganti misal random.seed(0) kalau mau map selalu sama

    W, H = 640, 640
    print("Generate peta random 640x640 dengan obstacle non-overlap (sebagian kecil overlap diizinkan)...")

    vis_img, occ, start, goal = generate_random_map(W, H, n_obstacles=18)
    print(f"Start: {start}, Goal: {goal}")

    # Canvas untuk ditampilkan (numpy)
    canvas = np.array(vis_img)

    # Setup matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    im_handle = ax.imshow(canvas)
    ax.set_title("Hybrid A* + RRT* pada Peta Random (Obstacles Non-Overlap-ish)")
    ax.axis("off")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    # Callback untuk planner
    def callback_draw_segment(seg):
        x1, y1, x2, y2 = seg
        # warna pohon RRT (abu-abu)
        draw_line_on_canvas(canvas, x1, y1, x2, y2, (120, 120, 120))
        im_handle.set_data(canvas)
        plt.pause(ANIM_DELAY)

    def callback_draw_best(path_pts: List[Tuple[int, int]]):
        # warnai path terbaik (kuning)
        for i in range(len(path_pts) - 1):
            x1, y1 = path_pts[i]
            x2, y2 = path_pts[i + 1]
            draw_line_on_canvas(canvas, x1, y1, x2, y2, (255, 255, 0))
        im_handle.set_data(canvas)
        plt.pause(0.1)

    # Jalankan planner
    planner = RRTStar(
        occ,
        start,
        goal,
        step_len=32,
        goal_sample_rate=0.05,
        neighbor_radius=60,
        max_iter=4000,
        narrow_bias_rate=0.12,
        bridge_sigma=10.0,
        bridge_trials=5,
        fixed_step=True,
        min_sample_sep=14,
        gamma_rewire=90.0,
        astar_guided_rate=0.30,  # Hybrid: A*-guided sampling
    )

    print("Mulai RRT* + Hybrid A* ...")
    t0 = time.time()
    path, iters = planner.plan(
        callback_draw_segment=callback_draw_segment,
        callback_draw_best=callback_draw_best
    )
    elapsed = time.time() - t0

    if path is None:
        print("Path TIDAK ditemukan.")
    else:
        print("Path ditemukan!")
    print(f"Iterasi: {iters}/{planner.max_iter}")
    print(f"Total node: {len(planner.nodes)}")
    print(f"Waktu   : {elapsed:.3f} detik")

    # Tahan window sampai ditutup
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
