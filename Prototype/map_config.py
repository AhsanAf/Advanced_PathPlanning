# map_config.py
# Skenario: "The Long Narrow Corridor" (Mimpi Buruk RRT Polos)

class MapConfig:
    def __init__(self):
        self.min_rand = 0
        self.max_rand = 100
        
        self.start_pos = [10, 50]
        self.goal_pos = [90, 50]
        
        self.obstacle_list = [
            # --- BLOK RAKSASA PEMBENTUK TEROWONGAN ---
            # Kita membuat lorong horisontal dari x=30 sampai x=70
            # Lorong ini ada di y=49 s/d y=51 (Hanya lebar 2 unit!)
            
            # 1. Blok Bawah (Menutup dari lantai sampai bibir bawah lorong)
            # x=30 s/d x=70, y=0 s/d y=49
            (30, 0, 40, 49),

            # 2. Blok Atas (Menutup dari bibir atas lorong sampai tinggi)
            # x=30 s/d x=70, y=51 s/d y=85
            # Kita sisakan celah LEBAR di paling atas (y=85 s/d y=100)
            (30, 51, 40, 34),

            # --- PENGHALANG VERTIKAL (AGAR TIDAK CURANG) ---
            # Memaksa robot benar-benar masuk lorong atau memutar jauh ke atas
            # Tidak bisa lewat bawah
            (30, 0, 5, 49),
            (65, 0, 5, 49),

            # --- DINDING LUAR ---
            (-5, -5, 110, 5),   # Bawah
            (-5, 100, 110, 5),  # Atas
            (-5, 0, 5, 100),    # Kiri
            (100, 0, 5, 100)    # Kanan
        ]

    def get_obstacles(self):
        return self.obstacle_list