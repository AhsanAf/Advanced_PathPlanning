#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic RRT with Goal-Bias and Rewire (RRT*) + Tkinter UI
- Input Gambar: choose image from disk (PNG/JPG). Colors:
    - #0000FF (blue)  : START
    - #FF0000 (red)   : GOAL
    - #000000 (black) : OBSTACLE
- Start: run RRT* with live animation on right canvas
- Reset: clear the session (keeps loaded image)
- Save: save current result view (right pane) to PNG
- Import: placeholder (no-op for now)
- Exit: close app
Left box  : displays original input image
Right box : displays RRT growth animation and final path (if found)
"""
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk
from tkinter import filedialog, messagebox

# ======= Pengaturan animasi =======
ANIM_DELAY = 0.005  # detik jeda per segmen (ubah sesuai selera)

# -------------------- RRT* core --------------------

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
        step_len: int = 10,
        goal_sample_rate: float = 0.1,  # goal bias
        neighbor_radius: int = 25,
        max_iter: int = 5000,
    ):
        """
        occ_map: HxW boolean array; True where obstacle, False free.
        start, goal: (x, y) pixel coordinates.
        """
        self.occ = occ_map
        self.h, self.w = occ_map.shape
        self.start = Node(*start, parent=None, cost=0.0)
        self.goal = Node(*goal, parent=None, cost=0.0)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.neighbor_radius = neighbor_radius
        self.max_iter = max_iter

        self.nodes: List[Node] = [self.start]

    def _is_free(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h and (not self.occ[y, x])

    def _sample(self) -> Tuple[int, int]:
        # goal-bias sampling
        if random.random() < self.goal_sample_rate:
            return self.goal.x, self.goal.y
        return random.randint(0, self.w - 1), random.randint(0, self.h - 1)

    def _nearest(self, x: int, y: int) -> int:
        dists = [(i, (n.x - x) ** 2 + (n.y - y) ** 2) for i, n in enumerate(self.nodes)]
        i_min = min(dists, key=lambda t: t[1])[0]
        return i_min

    def _steer(self, from_node: Node, to_point: Tuple[int, int]) -> Tuple[int, int]:
        fx, fy = from_node.x, from_node.y
        tx, ty = to_point
        dx = tx - fx
        dy = ty - fy
        dist = math.hypot(dx, dy)
        if dist == 0:
            return fx, fy
        scale = min(self.step_len / dist, 1.0)
        nx = int(round(fx + dx * scale))
        ny = int(round(fy + dy * scale))
        return nx, ny

    def _collision_free_segment(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Bresenham-style check between points inclusive."""
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
        idxs = []
        for i, n in enumerate(self.nodes):
            if (n.x - x) ** 2 + (n.y - y) ** 2 <= r2:
                idxs.append(i)
        return idxs

    def _cost(self, idx: int) -> float:
        return self.nodes[idx].cost

    def plan(self, callback_draw_segment=None, callback_draw_best=None) -> Optional[List[Tuple[int, int]]]:
        """Run RRT*; callbacks are used for animation drawing on UI.
        Returns path as list of (x,y) from start to goal if found.
        """
        for _ in range(self.max_iter):
            rx, ry = self._sample()
            nearest_idx = self._nearest(rx, ry)
            new_x, new_y = self._steer(self.nodes[nearest_idx], (rx, ry))

            if not self._is_free(new_x, new_y):
                continue
            if not self._collision_free_segment((self.nodes[nearest_idx].x, self.nodes[nearest_idx].y), (new_x, new_y)):
                continue

            # Choose parent (RRT*): minimal total cost
            new_cost_best = self._cost(nearest_idx) + math.hypot(new_x - self.nodes[nearest_idx].x,
                                                                 new_y - self.nodes[nearest_idx].y)
            new_parent = nearest_idx
            neighbor_ids = self._near_indices(new_x, new_y, self.neighbor_radius)
            for j in neighbor_ids:
                cand = self.nodes[j]
                if self._collision_free_segment((cand.x, cand.y), (new_x, new_y)):
                    cand_cost = self._cost(j) + math.hypot(new_x - cand.x, new_y - cand.y)
                    if cand_cost < new_cost_best:
                        new_cost_best = cand_cost
                        new_parent = j

            self.nodes.append(Node(new_x, new_y, parent=new_parent, cost=new_cost_best))
            new_idx = len(self.nodes) - 1

            # Rewire neighbors (RRT*)
            for j in neighbor_ids:
                if j == new_idx:
                    continue
                cand = self.nodes[j]
                new_cost = self._cost(new_idx) + math.hypot(cand.x - new_x, cand.y - new_y)
                if new_cost + 1e-6 < self._cost(j) and self._collision_free_segment((new_x, new_y), (cand.x, cand.y)):
                    self.nodes[j].parent = new_idx
                    self.nodes[j].cost = new_cost

            # Draw incremental edge
            if callback_draw_segment is not None:
                parent = self.nodes[new_parent]
                callback_draw_segment((parent.x, parent.y, new_x, new_y))

            # Check if we can connect to goal
            if self._collision_free_segment((new_x, new_y), (self.goal.x, self.goal.y)):
                self.nodes.append(Node(self.goal.x, self.goal.y, parent=new_idx,
                                       cost=self._cost(new_idx) + math.hypot(self.goal.x - new_x, self.goal.y - new_y)))
                goal_idx = len(self.nodes) - 1
                path = []
                cur = goal_idx
                while cur is not None:
                    n = self.nodes[cur]
                    path.append((n.x, n.y))
                    cur = n.parent
                path.reverse()
                if callback_draw_best is not None:
                    callback_draw_best(path)
                return path

        return None

# -------------------- Utilities --------------------

def _centroid_of_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    return (cx, cy)

def find_start_goal_from_image(img: Image.Image, tol: int = 50) -> Tuple[Optional[Tuple[int,int]], Optional[Tuple[int,int]]]:
    """
    Tolerant color detection.
    - Start: near blue (#0000FF)
    - Goal : near red  (#FF0000)
    We allow JPEG artifacts/anti-aliasing by using Euclidean distance in RGB.
    """
    arr = np.array(img.convert("RGB")).astype(np.int16)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Euclidean distance to target colors
    dist_blue = np.sqrt((r-0)**2 + (g-0)**2 + (b-255)**2)
    dist_red  = np.sqrt((r-255)**2 + (g-0)**2 + (b-0)**2)
    blue_mask = dist_blue <= tol
    red_mask  = dist_red  <= tol
    start = _centroid_of_mask(blue_mask)
    goal  = _centroid_of_mask(red_mask)
    return start, goal

def image_to_occ(img: Image.Image, black_thresh: int = 60) -> np.ndarray:
    """Obstacle mask from near-black pixels (handles JPEG)."""
    arr = np.array(img.convert("RGB")).astype(np.int16)
    obs = (arr[:,:,0] < black_thresh) & (arr[:,:,1] < black_thresh) & (arr[:,:,2] < black_thresh)
    return obs

# -------------------- Tkinter UI --------------------

class RRTApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RRT* (Goal-Bias + Rewire)")
        self.root.geometry("1200x720")
        self.root.configure(bg="white")

        self.loaded_image: Optional[Image.Image] = None
        self.loaded_path: Optional[str] = None
        self.left_photo = None  # keep references
        self.right_image: Optional[Image.Image] = None  # used for saving
        self.right_bg_photo = None

        # Top buttons (left -> right)
        top = tk.Frame(root, bg="white")
        top.pack(side=tk.TOP, fill=tk.X, padx=20, pady=12)

        self.btn_input = tk.Button(top, text="Input gambar", command=self.on_input, width=15)
        self.btn_start = tk.Button(top, text="Start", command=self.on_start, width=10)
        self.btn_reset = tk.Button(top, text="Reset", command=self.on_reset, width=10)
        self.btn_save  = tk.Button(top, text="Save", command=self.on_save, width=10)
        self.btn_import= tk.Button(top, text="Import", command=self.on_import, width=10)
        self.btn_exit  = tk.Button(top, text="Exit", command=self.on_exit, width=10)

        for b in [self.btn_input, self.btn_start, self.btn_reset, self.btn_save, self.btn_import, self.btn_exit]:
            b.pack(side=tk.LEFT, padx=10)

        # Two panels
        main = tk.Frame(root, bg="white")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.left_canvas = tk.Canvas(main, bg="white", highlightthickness=4, highlightbackground="#0D47A1")
        self.right_canvas = tk.Canvas(main, bg="white", highlightthickness=4, highlightbackground="#0D47A1")

        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Animation helpers
        self.anim_lines = []  # list of canvas line ids
        self.anim_path = []   # path line ids
        self.rrt_running = False

    # ---------- Button handlers ----------

    def on_input(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar (Start: biru, Goal: merah, Obstacles: hitam)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuka gambar:\n{e}")
            return
        self.loaded_image = img
        self.loaded_path = path
        self._draw_left_image(img)
        self._reset_right_panel(clear_image=True)

    def on_start(self):
        if self.loaded_image is None:
            messagebox.showwarning("Info", "Silakan pilih gambar terlebih dahulu.")
            return
        if self.rrt_running:
            return

        start, goal = find_start_goal_from_image(self.loaded_image)
        if start is None or goal is None:
            messagebox.showerror("Error", "Start (#0000FF) atau Goal (#FF0000) tidak ditemukan pada gambar.")
            return

        occ = image_to_occ(self.loaded_image)
        self._prepare_right_background()
        self.rrt_running = True

        # ---- Mapping gambar -> kanvas (presisi, pakai scale + offset) ----
        w_img, h_img = self.loaded_image.size
        w_can = max(self.right_canvas.winfo_width(), 10)
        h_can = max(self.right_canvas.winfo_height(), 10)
        scale = min(w_can / w_img, h_can / h_img)
        sx = sy = scale
        ox = (w_can - w_img * scale) / 2.0
        oy = (h_can - h_img * scale) / 2.0

        def draw_segment(seg):
            x1, y1, x2, y2 = seg
            cx1, cy1 = x1 * sx + ox, y1 * sy + oy
            cx2, cy2 = x2 * sx + ox, y2 * sy + oy
            line_id = self.right_canvas.create_line(cx1, cy1, cx2, cy2, width=1)
            self.anim_lines.append(line_id)
            self.right_canvas.update()
            if ANIM_DELAY > 0:
                time.sleep(ANIM_DELAY)

        def draw_best(path_pts):
            # clear previous best path
            for lid in self.anim_path:
                self.right_canvas.delete(lid)
            self.anim_path.clear()
            # draw thick line for best path
            for i in range(len(path_pts)-1):
                x1, y1 = path_pts[i]
                x2, y2 = path_pts[i+1]
                lid = self.right_canvas.create_line(x1*sx+ox, y1*sy+oy, x2*sx+ox, y2*sy+oy, width=3)
                self.anim_path.append(lid)
            self.right_canvas.update()

        def run_planner():
            planner = RRTStar(occ, start, goal,
                              step_len=10, goal_sample_rate=0.25,
                              neighbor_radius=25, max_iter=8000)
            path = planner.plan(callback_draw_segment=draw_segment, callback_draw_best=draw_best)
            self.rrt_running = False
            if path is None:
                messagebox.showinfo("Hasil", "Path tidak ditemukan.")
            else:
                messagebox.showinfo("Hasil", "Path ditemukan!")
            # save the right canvas snapshot for "Save"
            self.right_image = self._snapshot_right_canvas()

        # Run planner tanpa membekukan UI
        self.root.after(10, run_planner)

    def on_reset(self):
        self._reset_right_panel(clear_image=True)

    def on_save(self):
        if self.right_image is None:
            self.right_image = self._snapshot_right_canvas()
        if self.right_image is None:
            messagebox.showwarning("Info", "Tidak ada hasil untuk disimpan.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Simpan hasil",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not out_path:
            return
        try:
            self.right_image.save(out_path)
            messagebox.showinfo("Berhasil", f"Tersimpan: {out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan:\n{e}")

    def on_import(self):
        messagebox.showinfo("Import", "Fungsi Import belum diimplementasikan.")

    def on_exit(self):
        self.root.destroy()

    # ---------- Drawing helpers ----------

    def _draw_left_image(self, img: Image.Image):
        # Fit to left canvas size while maintaining aspect
        self.root.update_idletasks()
        cw = max(self.left_canvas.winfo_width(), 10)
        ch = max(self.left_canvas.winfo_height(), 10)
        imw, imh = img.size
        scale = min(cw / imw, ch / imh)
        new_size = (max(1, int(imw * scale)), max(1, int(imh * scale)))
        disp = img.resize(new_size, Image.NEAREST)
        self.left_photo = ImageTk.PhotoImage(disp)
        self.left_canvas.delete("all")
        self.left_canvas.create_image(cw // 2, ch // 2, image=self.left_photo, anchor="center")

    def _prepare_right_background(self):
        # clear and draw the map (with obstacles) as bitmap on the right canvas
        self.right_canvas.delete("all")
        cw = max(self.right_canvas.winfo_width(), 10)
        ch = max(self.right_canvas.winfo_height(), 10)
        self.right_canvas.create_rectangle(0, 0, cw, ch, fill="white", outline="")
        if self.loaded_image is not None:
            imw, imh = self.loaded_image.size
            scale = min(cw / imw, ch / imh)
            new_size = (max(1, int(imw * scale)), max(1, int(imh * scale)))
            disp = self.loaded_image.resize(new_size, Image.NEAREST)
            self.right_bg_photo = ImageTk.PhotoImage(disp)
            # center image with offset (anchor nw + offset)
            ox = (cw - new_size[0]) // 2
            oy = (ch - new_size[1]) // 2
            self.right_canvas.create_image(ox, oy, image=self.right_bg_photo, anchor="nw")
        self.anim_lines.clear()
        self.anim_path.clear()
        self.right_image = None

    def _reset_right_panel(self, clear_image: bool = False):
        self.right_canvas.delete("all")
        self.anim_lines.clear()
        self.anim_path.clear()
        if clear_image:
            self.right_image = None

    def _snapshot_right_canvas(self) -> Optional[Image.Image]:
        # Save canvas to an image by drawing it into a PIL image (vector-only elements drawn by Canvas).
        try:
            cw = max(self.right_canvas.winfo_width(), 10)
            ch = max(self.right_canvas.winfo_height(), 10)
            img = Image.new("RGB", (cw, ch), "white")
            draw = ImageDraw.Draw(img)
            for lid in self.right_canvas.find_all():
                coords = self.right_canvas.coords(lid)
                item_type = self.right_canvas.type(lid)
                if item_type == "line" and len(coords) >= 4:
                    draw.line(coords, fill=(0,0,0), width=2)
                elif item_type == "oval" and len(coords) == 4:
                    draw.ellipse(coords, fill=(0,0,0))
                elif item_type == "rectangle" and len(coords) == 4:
                    draw.rectangle(coords, fill=(255,255,255), outline=None)
            return img
        except Exception:
            return None

def main():
    root = tk.Tk()
    app = RRTApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
