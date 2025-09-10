import os
import json
import sys
import threading
import math
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
try:
    from ttkbootstrap import Style  # type: ignore
except Exception:
    Style = None  # optional

# Local imports (run from repo root)
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from collections import defaultdict, Counter
from catan_plus import make_standard_board, Game
from starter_eval import simulate, build_board, build_players


RESOURCES = ["brick", "wood", "wool", "grain", "ore", None]
RES_COLORS = {
    None: "#353a4e",
    "brick": "#b55a3c",
    "wood": "#1c7b46",
    "wool": "#8bcf63",
    "grain": "#d7b44a",
    "ore": "#6f6f7e",
}
PLAYER_COLORS = ["#5bd4ff", "#ff95e9", "#ffd24d", "#78f596"]

# Number tokens per standard spiral set
TOKEN_ORDER = [5,2,6,3,8,10,9,12,11,4,8,10,9,4,5,6,3,11]
TOKEN_LIMITS = Counter(TOKEN_ORDER)

# Editor-only sentinel for empty hex (distinct from desert=None)
UNSET = "__unset__"
RESOURCE_LIMITS = {None: 1, "brick": 3, "wood": 4, "wool": 4, "grain": 4, "ore": 3}
# Port limits for a standard 9-port setup: 4 generic, 1 per resource
PORT_LIMITS = {"generic": 4, "brick": 1, "wood": 1, "wool": 1, "grain": 1, "ore": 1}


class BoardEditorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Catan Board Editor & Simulator")
        self.root.geometry("1280x820")

        # Model
        self.board = make_standard_board(seed=0)
        # Start with an empty board: no resources, numbers, or ports
        self.hex_resources = {hid: UNSET for hid in self.board.hexes.keys()}
        self.hex_numbers = {hid: None for hid in self.board.hexes.keys()}
        self.board.ports = {}
        self.robber_hex = 0
        self.num_players = tk.IntVar(value=2)
        self.player_names: list[tk.StringVar] = [tk.StringVar(value=f"Player {i+1}") for i in range(4)]
        self.players = [
            {"settlements": set(), "cities": set(), "roads": set()}
            for _ in range(4)
        ]

        # UI state
        self.active_player = tk.IntVar(value=0)
        self.games_var = tk.StringVar(value="500")
        self.turns_var = tk.StringVar(value="300")

        # Layout
        top = ttk.Frame(root)
        top.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(top, text="Players:").pack(side=tk.LEFT)
        tk.Spinbox(top, from_=2, to=4, width=4, textvariable=self.num_players, command=self.on_num_players_change).pack(side=tk.LEFT, padx=(4,10))
        for i in range(4):
            ttk.Entry(top, textvariable=self.player_names[i], width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="#Games:").pack(side=tk.LEFT, padx=(16,4))
        ttk.Entry(top, textvariable=self.games_var, width=8).pack(side=tk.LEFT)
        ttk.Label(top, text="Max Turns:").pack(side=tk.LEFT, padx=(8,4))
        ttk.Entry(top, textvariable=self.turns_var, width=8).pack(side=tk.LEFT)
        self.run_btn = ttk.Button(top, text="Run Simulation", command=self.run_simulation, style="primary.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=8)
        # Single run button
        self.run_one_btn = ttk.Button(top, text="Run One Game", command=self.run_single_game)
        self.run_one_btn.pack(side=tk.LEFT)
        self.rand_btn = ttk.Button(top, text="Random Map", command=self.randomize_full_board)
        self.rand_btn.pack(side=tk.LEFT, padx=(4,0))
        # Strong bots toggle
        self.use_strong_bots = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Strong Bots", variable=self.use_strong_bots).pack(side=tk.LEFT, padx=8)
        # MCTS params controls
        ttk.Label(top, text="Beam:").pack(side=tk.LEFT)
        self.mcts_beam_var = tk.IntVar(value=6)
        tk.Spinbox(top, from_=1, to=64, width=4, textvariable=self.mcts_beam_var).pack(side=tk.LEFT, padx=(2,8))
        ttk.Label(top, text="Depth:").pack(side=tk.LEFT)
        self.mcts_depth_var = tk.IntVar(value=3)
        tk.Spinbox(top, from_=1, to=10, width=4, textvariable=self.mcts_depth_var).pack(side=tk.LEFT, padx=(2,8))
        # Toolbar save/load
        ttk.Button(top, text="Save", command=self.save_board).pack(side=tk.LEFT, padx=(8,0))
        ttk.Button(top, text="Open", command=self.load_board).pack(side=tk.LEFT, padx=(4,0))
        ttk.Button(top, text="From Image", command=self.import_from_image).pack(side=tk.LEFT, padx=(4,0))
        self.status = ttk.Label(top, text="Idle", anchor=tk.W)
        self.status.pack(side=tk.LEFT, padx=12)

        # Ensure initial window is wide enough to show toolbar fully
        try:
            self.root.update_idletasks()
            min_w = max(self.root.winfo_width(), top.winfo_reqwidth() + 24)
            min_h = max(self.root.winfo_height(), 720)
            self.root.minsize(min_w, min_h)
        except Exception:
            pass

        body = tk.PanedWindow(root, sashrelief=tk.RAISED, sashwidth=6)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left: canvas
        left = ttk.Frame(body)
        body.add(left)
        self.canvas = tk.Canvas(left, width=720, height=640, bg="#0e1320", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: palette and results
        right = ttk.Frame(body)
        body.add(right)
        bar = ttk.Frame(right)
        bar.pack(fill=tk.X)
        self.enforce_rules = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Enforce Catan Legality", variable=self.enforce_rules, command=self.redraw).pack(side=tk.LEFT)
        # Setup mode: restrict to 2 settlements + 2 roads per player, roads must touch a settlement
        self.setup_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Setup Mode (2 settlements + 2 roads)", variable=self.setup_mode, command=self.on_setup_mode_toggle).pack(side=tk.LEFT, padx=(8,0))
        # HUD toggle to show per-player cards and dice rolls
        self.show_hud = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show HUD", variable=self.show_hud, command=self.redraw).pack(side=tk.LEFT, padx=(8,0))
        ttk.Label(bar, text="  Active Player:").pack(side=tk.LEFT)
        self.player_buttons: list[tk.Radiobutton] = []
        def rebuild_player_buttons(*_):
            for b in self.player_buttons:
                b.destroy()
            self.player_buttons.clear()
            for i in range(self.num_players.get()):
                rb = ttk.Radiobutton(bar, text=str(i+1), value=i, variable=self.active_player)
                rb.pack(side=tk.LEFT)
                self.player_buttons.append(rb)
        rebuild_player_buttons()
        self.num_players.trace_add('write', lambda *a: rebuild_player_buttons())

        # Prepare counters dicts before palette builds labels
        self.resource_count_labels: dict[object, tk.Label] = {}
        self.number_count_labels: dict[int, tk.Label] = {}
        # Prepare counters dicts before palette builds labels
        self.resource_count_labels: dict[object, tk.Label] = {}
        self.number_count_labels: dict[int, tk.Label] = {}
        self.port_count_labels: dict[str, tk.Label] = {}
        self.build_palette(right)

        # Setup progress block
        self.setup_prog = ttk.LabelFrame(right, text="Setup Progress")
        self.setup_prog.pack(fill=tk.X, pady=(8,0))
        self.setup_progress_labels: list[ttk.Label] = []
        for _ in range(4):
            lbl = ttk.Label(self.setup_prog, text="", anchor=tk.W, justify=tk.LEFT)
            lbl.pack(fill=tk.X)
            self.setup_progress_labels.append(lbl)

        ttk.Label(right, text="Results").pack(anchor=tk.W, pady=(8,0))
        self.results_txt = tk.Text(right, height=16, relief=tk.FLAT)
        self.results_txt.pack(fill=tk.BOTH, expand=True)

        # --- Replay controls (play the full game move-by-move) ---
        self.replay_trace: list[dict] = []
        self.replay_logs: list[str] = []
        self.replay_index = tk.IntVar(value=0)
        self.replay_playing = False

        replay = ttk.LabelFrame(right, text="Replay")
        replay.pack(fill=tk.BOTH, expand=False, pady=(8,0))
        ctrls = ttk.Frame(replay); ctrls.pack(fill=tk.X, pady=(4,4))
        self.capture_btn = ttk.Button(ctrls, text="Simulate & Capture", command=self.capture_game_to_replay)
        self.capture_btn.pack(side=tk.LEFT)
        ttk.Separator(ctrls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.first_btn = ttk.Button(ctrls, text="|<", width=3, command=self.replay_first)
        self.prev_btn = ttk.Button(ctrls, text="<", width=3, command=self.replay_prev)
        self.play_btn = ttk.Button(ctrls, text="Play", width=5, command=self.replay_toggle_play)
        self.next_btn = ttk.Button(ctrls, text=">", width=3, command=self.replay_next)
        self.last_btn = ttk.Button(ctrls, text=">|", width=3, command=self.replay_last)
        for b in (self.first_btn, self.prev_btn, self.play_btn, self.next_btn, self.last_btn):
            b.pack(side=tk.LEFT, padx=2)
        # Slider + status
        sl = ttk.Frame(replay); sl.pack(fill=tk.X)
        ttk.Label(sl, text="Step:").pack(side=tk.LEFT)
        self.replay_scale = tk.Scale(sl, from_=0, to=0, orient=tk.HORIZONTAL, showvalue=False,
                                     variable=self.replay_index, command=lambda _=None: self.replay_goto(self.replay_index.get()))
        self.replay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.replay_status = ttk.Label(sl, text="0/0")
        self.replay_status.pack(side=tk.LEFT)
        # Current note and per-player cards/VP
        self.replay_note = ttk.Label(replay, text="", anchor=tk.W, justify=tk.LEFT)
        self.replay_note.pack(fill=tk.X, padx=2)
        self.replay_info = tk.Text(replay, height=8, relief=tk.FLAT)
        self.replay_info.pack(fill=tk.BOTH, expand=False)

        # Bindings
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_drop)
        self.root.bind("<B1-Motion>", self.on_global_drag)
        self.root.bind("<ButtonRelease-1>", self.on_global_drop)
        # Zoom / Pan
        self.scale = 1.0
        self.pan = [0.0, 0.0]
        self._pan_start = None
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)

        # Caches
        self.node_px: dict[int, tuple[float, float]] = {}
        self.edge_mid: dict[tuple[int, int], tuple[float, float]] = {}
        self.hex_center: dict[int, tuple[float, float]] = {}
        self.dragging = None  # {kind,data,pos,target,chip_bg}

        # Replay/HUD state
        self.replay_trace: list[dict] = []
        self.replay_logs: list[str] = []
        self.replay_index = tk.IntVar(value=0)
        self.replay_playing = False
        self.hud_players: list[dict] = []  # per-frame player dicts
        self.hud_roll: object | None = None
        self.hud_current: int | None = None

        # Track settlement placement order for each player (used to identify "second" settlement)
        self.setup_settlement_order: list[list[int]] = [[] for _ in range(4)]
        self.setup_road_assigned: list[set[int]] = [set() for _ in range(4)]
        self.second_settlement: dict[int,int] = {}

        self.redraw()

    # ---------- UI construction ----------
    def build_palette(self, parent: tk.Frame):
        pal = ttk.LabelFrame(parent, text="Palette (drag and drop)")
        pal.pack(fill=tk.X, pady=(8,0))

        # Pieces
        rowA = ttk.Frame(pal); rowA.pack(fill=tk.X, pady=2)
        ttk.Label(rowA, text="Pieces:").pack(side=tk.LEFT)
        self._make_draggable_chip(rowA, text="Settlement", bg=lambda: PLAYER_COLORS[self.active_player.get()%4], kind="piece", data="settlement")
        self._make_draggable_chip(rowA, text="City", bg=lambda: PLAYER_COLORS[self.active_player.get()%4], kind="piece", data="city")
        self._make_draggable_chip(rowA, text="Road", bg=lambda: PLAYER_COLORS[self.active_player.get()%4], kind="piece", data="road")
        self._make_draggable_chip(rowA, text="Erase Piece", bg="#402b2b", fg="#ffd0d0", kind="piece_erase", data=None)

        # Resources
        rowB = ttk.Frame(pal); rowB.pack(fill=tk.X, pady=2)
        ttk.Label(rowB, text="Resources:").pack(side=tk.LEFT)
        for r in ["brick","wood","wool","grain","ore"]:
            self._make_draggable_chip(rowB, text=r.capitalize(), bg=RES_COLORS[r], kind="resource", data=r)
        self._make_draggable_chip(rowB, text="Desert", bg=RES_COLORS[None], kind="resource", data=None)
        self._make_draggable_chip(rowB, text="Robber", bg="#2b3045", fg="#e24a67", kind="robber", data=None)

        # Numbers
        rowC = ttk.Frame(pal); rowC.pack(fill=tk.X, pady=2)
        ttk.Label(rowC, text="Numbers:").pack(side=tk.LEFT)
        for n in [2,3,4,5,6,8,9,10,11,12]:
            self._make_draggable_chip(rowC, text=str(n), bg="#f2d180" if n in (6,8) else "#d7d7e0", kind="number", data=n)

        # Ports
        rowD = ttk.Frame(pal); rowD.pack(fill=tk.X, pady=2)
        ttk.Label(rowD, text="Ports:").pack(side=tk.LEFT)
        self._make_draggable_chip(rowD, text="3:1", bg="#24395e", kind="port", data={"type":"generic","rate":3,"resource":None})
        for r in ["brick","wood","wool","grain","ore"]:
            self._make_draggable_chip(rowD, text=f"2:1 {r[0].upper()}", bg="#2c354f", kind="port", data={"type":"resource","rate":2,"resource":r})
        self._make_draggable_chip(rowD, text="Port Erase", bg="#402b2b", fg="#ffd0d0", kind="port_erase", data=None)

        # Resource tile counts
        rc = ttk.Frame(pal); rc.pack(fill=tk.X, pady=(6,0))
        ttk.Label(rc, text="Tiles left:").pack(side=tk.LEFT)
        for r in [None, "brick", "wood", "wool", "grain", "ore"]:
            lbl = tk.Label(rc, text="0", bg=RES_COLORS.get(r, "#2a3046"), fg="#101425" if r == "grain" else "#e6eefb", padx=6)
            lbl.pack(side=tk.LEFT, padx=2)
            self.resource_count_labels[r] = lbl

        # Number token counts
        nc = ttk.Frame(pal); nc.pack(fill=tk.X, pady=(4,0))
        ttk.Label(nc, text="Numbers left:").pack(side=tk.LEFT)
        for n in [2,3,4,5,6,8,9,10,11,12]:
            lbl = tk.Label(nc, text=str(TOKEN_LIMITS[n]), bg="#f2d180" if n in (6,8) else "#d7d7e0", fg="#101425", padx=6)
            lbl.pack(side=tk.LEFT, padx=2)
            self.number_count_labels[n] = lbl

        # Ports left counters
        pc = ttk.Frame(pal); pc.pack(fill=tk.X, pady=(4,0))
        ttk.Label(pc, text="Ports left:").pack(side=tk.LEFT)
        for key, label_text in [("generic","3:1"), ("brick","B"), ("wood","W"), ("wool","L"), ("grain","G"), ("ore","O")]:
            bg = "#24395e" if key=="generic" else "#2c354f"
            lbl = tk.Label(pc, text=str(PORT_LIMITS[key]), bg=bg, fg="#bcd", padx=6)
            lbl.pack(side=tk.LEFT, padx=2)
            self.port_count_labels[key] = lbl

    def _make_draggable_chip(self, parent, text, kind, data, bg="#2a3046", fg="#e6eefb"):
        def resolve_bg():
            return bg() if callable(bg) else bg
        lbl = tk.Label(parent, text=text, bg=resolve_bg(), fg=fg, relief=tk.RIDGE, bd=1, padx=6, pady=2)
        lbl.pack(side=tk.LEFT, padx=3)
        def start_drag(_):
            self.dragging = {"kind": kind, "data": data, "pos": None, "target": None, "chip_bg": resolve_bg()}
        lbl.bind("<Button-1>", start_drag)

    # ---------- Geometry & Drawing ----------
    def compute_transform(self, w: int, h: int):
        nodes = self.board.nodes
        xs = [p[0] for p in nodes.values()]
        ys = [p[1] for p in nodes.values()]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        bw, bh = (maxx-minx), (maxy-miny)
        pad = 40
        base = min((w-2*pad)/bw, (h-2*pad)/bh) if bw>0 and bh>0 else 1.0
        scale = base * (self.scale if hasattr(self, 'scale') else 1.0)
        ox = (w - scale*bw)/2 - scale*minx + (self.pan[0] if hasattr(self,'pan') else 0)
        oy = (h - scale*bh)/2 - scale*miny + (self.pan[1] if hasattr(self,'pan') else 0)
        def P(pt):
            return (ox + scale*pt[0], oy + scale*pt[1])
        return P

    def redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 720
        h = self.canvas.winfo_height() or 640
        P = self.compute_transform(w, h)
        # caches
        self.node_px = {nid: P(pt) for nid, pt in self.board.nodes.items()}
        self.edge_mid = { (min(u,v),max(u,v)): ((self.node_px[u][0]+self.node_px[v][0])/2.0,
                                               (self.node_px[u][1]+self.node_px[v][1])/2.0)
                          for (u,v) in self.board.edges}
        self.hex_center = {}

        # hexes
        for hid, hx in self.board.hexes.items():
            pts = [self.node_px[n] for n in hx.nodes]
            rv = self.hex_resources.get(hid)
            color = RES_COLORS.get(rv if rv is not UNSET else None, "#2a3046")
            self.canvas.create_polygon(*sum(([x,y] for (x,y) in pts), []), fill=color, outline="#101425", width=1)
            cx = sum(p[0] for p in pts)/6.0; cy = sum(p[1] for p in pts)/6.0
            self.hex_center[hid] = (cx, cy)
            num = self.hex_numbers.get(hid)
            if num is not None:
                r = 12; hot = (num in (6,8))
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#f2d180" if hot else "#d7d7e0", outline="#101425")
                self.canvas.create_text(cx, cy, text=str(num), fill="#101425", font=("Segoe UI", 10, "bold"))
            if hid == self.robber_hex:
                r2 = 7
                self.canvas.create_oval(cx-r2, cy-r2, cx+r2, cy+r2, fill="#2b3045", outline="#e24a67", width=2)

        # roads
        for pi in range(self.num_players.get()):
            col = PLAYER_COLORS[pi % len(PLAYER_COLORS)]
            for (u,v) in sorted(self.players[pi]["roads"]):
                up = self.node_px[u]; vp = self.node_px[v]
                self.canvas.create_line(up[0],up[1],vp[0],vp[1], width=5, fill=col)

        # settlements & cities
        for pi in range(self.num_players.get()):
            col = PLAYER_COLORS[pi % len(PLAYER_COLORS)]
            for nid in self.players[pi]["settlements"]:
                x,y = self.node_px[nid]
                # House-shaped settlement: larger base + roof for readability
                base_w = 18.0
                base_h = 12.0
                roof_h = 10.0
                pts = [
                    (x - base_w/2, y + base_h/2),  # bottom-left
                    (x + base_w/2, y + base_h/2),  # bottom-right
                    (x + base_w/2, y - base_h/2),  # top-right of base
                    (x,              y - base_h/2 - roof_h),  # roof peak
                    (x - base_w/2, y - base_h/2),  # top-left of base
                ]
                self.canvas.create_polygon(*sum(([px,py] for (px,py) in pts), []), fill=col, outline="#101425", width=1)
            for nid in self.players[pi]["cities"]:
                x, y = self.node_px[nid]
                # City: walled base with a central tower (distinct from settlement)
                wall_w = 22.0
                wall_h = 12.0
                tower_h = 10.0
                pts = [
                    (x - wall_w/2, y + wall_h/2),  # bottom-left
                    (x + wall_w/2, y + wall_h/2),  # bottom-right
                    (x + wall_w/2, y - wall_h/2),  # top-right of wall
                    (x + wall_w/6, y - wall_h/2),  # right step into tower
                    (x + wall_w/6, y - wall_h/2 - tower_h),  # tower top-right
                    (x - wall_w/6, y - wall_h/2 - tower_h),  # tower top-left
                    (x - wall_w/6, y - wall_h/2),  # left step out of tower
                    (x - wall_w/2, y - wall_h/2),  # top-left of wall
                ]
                self.canvas.create_polygon(*sum(([px,py] for (px,py) in pts), []), fill=col, outline="#101425", width=1)

        # interaction markers
        for nid,(x,y) in self.node_px.items():
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="#9fb3e4", outline="")
        for (u,v),(mx,my) in self.edge_mid.items():
            self.canvas.create_oval(mx-2, my-2, mx+2, my+2, fill="#8aa0d6", outline="")

        # ports
        for nid,port in self.board.ports.items():
            if nid not in self.node_px: continue
            x,y = self.node_px[nid]
            fill = '#24395e' if port.get('type')=='generic' else '#2c354f'
            self.canvas.create_oval(x-9,y-9,x+9,y+9, fill=fill, outline='#8aa0d6', width=1)
            label = '3:1' if port.get('type')=='generic' else '2:1'
            self.canvas.create_text(x, y-14, text=label, fill='#bcd', font=("Segoe UI", 8))
            if port.get('resource'):
                self.canvas.create_text(x, y+12, text=str(port['resource'])[0].upper(), fill='#bcd', font=("Segoe UI", 8))

        # drag ghost
        if self.dragging and self.dragging.get("pos"):
            kind = self.dragging["kind"]
            x,y = self.dragging["pos"]
            if kind == "piece":
                if self.dragging["data"] == "road":
                    e = self.nearest_edge(x,y)
                    if e:
                        u,v = e; up = self.node_px[u]; vp = self.node_px[v]
                        self.canvas.create_line(up[0],up[1],vp[0],vp[1], width=7, fill=self.dragging["chip_bg"], stipple="gray25")
                        self.dragging["target"] = ("edge", e)
                else:
                    nid = self.nearest_node(x,y)
                    if nid is not None:
                        xx,yy = self.node_px[nid]; s = 10 if self.dragging["data"]=="settlement" else 14
                        if self.dragging["data"] == "settlement":
                            # Preview house outline (larger) for settlement placement
                            base_w = 18.0
                            base_h = 12.0
                            roof_h = 10.0
                            pts = [
                                (xx - base_w/2, yy + base_h/2),
                                (xx + base_w/2, yy + base_h/2),
                                (xx + base_w/2, yy - base_h/2),
                                (xx,             yy - base_h/2 - roof_h),
                                (xx - base_w/2, yy - base_h/2),
                            ]
                            self.canvas.create_polygon(*sum(([px,py] for (px,py) in pts), []), outline=self.dragging["chip_bg"], width=3, fill="")
                        else:
                            # Preview city outline: walled base with central tower
                            wall_w = 22.0
                            wall_h = 12.0
                            tower_h = 10.0
                            pts = [
                                (xx - wall_w/2, yy + wall_h/2),
                                (xx + wall_w/2, yy + wall_h/2),
                                (xx + wall_w/2, yy - wall_h/2),
                                (xx + wall_w/6, yy - wall_h/2),
                                (xx + wall_w/6, yy - wall_h/2 - tower_h),
                                (xx - wall_w/6, yy - wall_h/2 - tower_h),
                                (xx - wall_w/6, yy - wall_h/2),
                                (xx - wall_w/2, yy - wall_h/2),
                            ]
                            self.canvas.create_polygon(*sum(([px,py] for (px,py) in pts), []), outline=self.dragging["chip_bg"], width=3, fill="")
                        self.dragging["target"] = ("node", nid)
            elif kind in ("resource","number","robber"):
                hid = self.nearest_hex(x,y)
                if hid is not None:
                    cx,cy = self.hex_center[hid]
                    if kind == "resource":
                        self.canvas.create_oval(cx-16,cy-16,cx+16,cy+16, outline="#e6eefb", width=2)
                    elif kind == "number":
                        self.canvas.create_oval(cx-14,cy-14,cx+14,cy+14, outline="#e6eefb", width=2)
                    else:
                        self.canvas.create_oval(cx-10,cy-10,cx+10,cy+10, outline="#e24a67", width=2)
                    self.dragging["target"] = ("hex", hid)
            elif kind in ("port","port_erase"):
                nid = self.nearest_border_node(x,y)
                if nid is not None:
                    xx,yy = self.node_px[nid]
                    self.canvas.create_oval(xx-10,yy-10,xx+10,yy+10, outline="#8aa0d6", width=2)
                    self.dragging["target"] = ("port", nid)
            elif kind == "piece_erase":
                nid = self.nearest_node(x,y); e = self.nearest_edge(x,y)
                if nid is not None:
                    xx,yy = self.node_px[nid]
                    self.canvas.create_oval(xx-8,yy-8,xx+8,yy+8, outline="#ff7b7b", width=2)
                    self.dragging["target"] = ("node", nid)
                if e is not None:
                    u,v = e; up = self.node_px[u]; vp = self.node_px[v]
                    self.canvas.create_line(up[0],up[1],vp[0],vp[1], width=6, fill="#ff7b7b")
                    self.dragging["target"] = ("edge", e)

        # update counters display
        self.update_counters(); self.update_setup_progress()

        # Draw HUD on top
        if getattr(self, 'show_hud', None) and self.show_hud.get():
            self.draw_hud()

    # ---------- HUD overlays ----------
    def draw_hud(self):
        w = self.canvas.winfo_width() or 720
        h = self.canvas.winfo_height() or 640
        pad = 10
        pw, ph = 250, 86
        corners = [
            (pad, pad),  # P1 top-left
            (w - pw - pad, pad),  # P2 top-right
            (w - pw - pad, h - ph - pad),  # P3 bottom-right
            (pad, h - ph - pad),  # P4 bottom-left
        ]
        # Resource abbreviation order
        order = ["brick","wood","wool","grain","ore"]
        abbrev = {"brick":"B","wood":"W","wool":"L","grain":"G","ore":"O"}
        N = self.num_players.get()
        for i in range(N):
            x0, y0 = corners[i]
            x1, y1 = x0 + pw, y0 + ph
            col = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            # Panel bg and border
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#1a2233", outline="#30425e", width=2)
            # Header with player color strip
            self.canvas.create_rectangle(x0, y0, x1, y0+18, fill=col, outline="")
            name = self.player_names[i].get() or f"Player {i+1}"
            self.canvas.create_text(x0+6, y0+9, text=f"P{i+1} {name}", anchor="w", fill="#101425", font=("Segoe UI", 9, "bold"))
            # Current turn indicator
            if self.hud_current is not None and i == int(self.hud_current):
                self.canvas.create_text(x1-6, y0+9, text="•", anchor="e", fill="#101425", font=("Segoe UI", 14, "bold"))
            # Body lines: VP and hand, plus dice roll at right side
            vp = 0; hvp = 0; hand = {}
            if 0 <= i < len(self.hud_players):
                p = self.hud_players[i]
                vp = int(p.get('vp', 0)); hvp = int(p.get('hidden_vp', 0))
                hand = p.get('hand', {}) or {}
            # VP line
            self.canvas.create_text(x0+8, y0+32, text=f"VP {vp}+{hvp}*", anchor="w", fill="#dfe7ff", font=("Segoe UI", 9))
            # Hand line
            parts = []
            for r in order:
                c = int(hand.get(r, 0))
                if c:
                    parts.append(f"{abbrev[r]}:{c}")
            hand_txt = ", ".join(parts) if parts else "Hand: -"
            # Prefix 'Hand: ' only once
            if parts:
                hand_txt = "Hand: " + hand_txt
            self.canvas.create_text(x0+8, y0+52, text=hand_txt, anchor="w", fill="#cbd6f4", font=("Segoe UI", 9))
            # Dice roll (if known)
            roll = self.hud_roll
            if roll is not None:
                self.canvas.create_text(x1-8, y0+52, text=f"Roll: {roll}", anchor="e", fill="#bfe27a", font=("Segoe UI", 10, "bold"))

    # ---------- Editing logic ----------
    def on_num_players_change(self):
        n = self.num_players.get()
        for i in range(n, 4):
            self.players[i]["settlements"].clear()
            self.players[i]["cities"].clear()
            self.players[i]["roads"].clear()
        self.redraw()

    def on_setup_mode_toggle(self):
        # Prevent turning off Setup Mode while any settlement lacks an adjacent road
        if not self.setup_mode.get() and self.enforce_rules.get():
            violations = []
            for i in range(self.num_players.get()):
                for nid in self.players[i]["settlements"]:
                    if not self.has_adjacent_road(i, nid):
                        violations.append((i, nid))
            if violations:
                self.setup_mode.set(True)
                self.toast("Finish setup: each settlement needs an adjacent road", "error")
        self.redraw()

    def nearest_node(self, x: float, y: float, max_dist: float=18.0):
        best = None; bd = max_dist
        for nid,(nx,ny) in self.node_px.items():
            d = math.hypot(nx-x, ny-y)
            if d < bd:
                bd = d; best = nid
        return best

    def nearest_edge(self, x: float, y: float, max_dist: float=16.0):
        best = None; bd = max_dist
        for e,(mx,my) in self.edge_mid.items():
            d = math.hypot(mx-x, my-y)
            if d < bd:
                bd = d; best = e
        return best

    def nearest_hex(self, x: float, y: float, max_dist: float=28.0):
        best = None; bd = max_dist
        for hid,(cx,cy) in self.hex_center.items():
            d = math.hypot(cx-x, cy-y)
            if d < bd:
                bd = d; best = hid
        return best

    def is_border_node(self, nid: int) -> bool:
        return len(self.board.adj.get(nid, ())) <= 2

    def nearest_border_node(self, x: float, y: float, max_dist: float=24.0):
        best = None; bd = max_dist
        for nid,(nx,ny) in self.node_px.items():
            if not self.is_border_node(nid):
                continue
            d = math.hypot(nx-x, ny-y)
            if d < bd:
                bd = d; best = nid
        return best

    # ---------- Legality helpers ----------
    def occupied_nodes(self):
        occ = set()
        for i in range(self.num_players.get()):
            occ |= self.players[i]["settlements"]
            occ |= self.players[i]["cities"]
        return occ

    def violates_distance_rule(self, nid: int) -> bool:
        occ = self.occupied_nodes()
        if nid in occ:
            return True
        for nb in self.board.adj.get(nid, ()): 
            if nb in occ:
                return True
        return False

    def player_connected_nodes(self, pi: int):
        owned = set(self.players[pi]["settlements"]) | set(self.players[pi]["cities"]) | set()
        roads = set(self.players[pi]["roads"]) | set()
        adj = defaultdict(set)
        for (u,v) in roads:
            adj[u].add(v); adj[v].add(u)
        seen = set(); stack = list(owned) + [n for e in roads for n in e]
        for start in stack:
            if start in seen: continue
            dq = [start]; seen.add(start)
            while dq:
                x = dq.pop(0)
                for y in adj.get(x, ()): 
                    if y not in seen:
                        seen.add(y); dq.append(y)
        return seen | owned

    def node_owner(self, nid: int):
        for i in range(self.num_players.get()):
            if nid in self.players[i]["settlements"] or nid in self.players[i]["cities"]:
                return i
        return None

    def has_adjacent_road(self, pi: int, nid: int) -> bool:
        for (u,v) in self.players[pi]["roads"]:
            if u == nid or v == nid:
                return True
        return False

    def pending_setup_settlement(self, pi: int):
        """Return the most recently placed settlement by player pi that has no road assigned yet, or None."""
        for nid in reversed(self.setup_settlement_order[pi]):
            if nid in self.players[pi]["settlements"] and nid not in self.setup_road_assigned[pi]:
                return nid
        return None

    def is_legal_settlement(self, pi: int, nid: int) -> bool:
        # setup-mode cap: only two settlements during initial placement
        if self.setup_mode.get() and len(self.players[pi]["settlements"]) >= 2:
            return False
        if len(self.players[pi]["settlements"]) >= 5:
            return False
        if self.violates_distance_rule(nid):
            return False
        if not self.enforce_rules.get():
            return True
        # During setup, allow placing the two initial settlements anywhere (distance rule already checked)
        # BUT enforce order: after placing a settlement, you must place an adjacent road before placing another settlement.
        if self.setup_mode.get():
            pending = self.pending_setup_settlement(pi)
            if pending is not None:
                return False
            return True
        # In regular play, a new settlement must be at the end of one of your roads
        for (u,v) in self.players[pi]["roads"]:
            if u == nid or v == nid:
                return True
        return False

    def is_legal_city(self, pi: int, nid: int) -> bool:
        if nid not in self.players[pi]["settlements"]:
            return False
        if len(self.players[pi]["cities"]) >= 4:
            return False
        return True

    def is_legal_road(self, pi: int, e: tuple[int,int]) -> bool:
        u,v = e
        if e not in self.board.edges:
            return False
        for i in range(self.num_players.get()):
            if e in self.players[i]["roads"]:
                return False
        # setup-mode cap: only two roads during initial placement
        if self.setup_mode.get() and len(self.players[pi]["roads"]) >= 2:
            return False
        if len(self.players[pi]["roads"]) >= 15:
            return False
        if not self.enforce_rules.get():
            return True
        # Block building a road that touches an opponent's settlement/city on any endpoint
        owner_u = self.node_owner(u)
        owner_v = self.node_owner(v)
        if (owner_u is not None and owner_u != pi) or (owner_v is not None and owner_v != pi):
            return False
        # In setup mode require road to touch the last unassigned settlement
        if self.setup_mode.get():
            # determine most recently placed settlement without an assigned road
            pending = self.pending_setup_settlement(pi)
            if pending is None:
                settlements = self.players[pi]["settlements"]
                return (u in settlements) or (v in settlements)
            return (u == pending) or (v == pending)
        # General connectivity rule otherwise: must connect to your frontier (roads or buildings)
        frontier = set(self.players[pi]["settlements"]) | set(self.players[pi]["cities"]) | set()
        for a,b in self.players[pi]["roads"]:
            frontier.add(a); frontier.add(b)
        return (u in frontier) or (v in frontier)

    # ---------- Drag handlers ----------
    def on_canvas_drag(self, event):
        if self.dragging is None:
            return
        self.dragging["pos"] = (event.x, event.y)
        self.redraw()

    def on_canvas_drop(self, event):
        if self.dragging is None:
            return
        drag = self.dragging; self.dragging = None
        target = drag.get("target")
        if not target:
            self.redraw(); return
        kind = drag["kind"]; data = drag["data"]; pi = self.active_player.get()
        if kind == "piece":
            if data == "road" and target[0] == "edge":
                e = target[1]
                if self.is_legal_road(pi, e):
                    self.players[pi]["roads"].add(e)
                    if self.enforce_rules.get() and self.setup_mode.get():
                        # mark assigned for the pending settlement if touched
                        pending = None
                        for nid in reversed(self.setup_settlement_order[pi]):
                            if nid not in self.setup_road_assigned[pi] and nid in self.players[pi]["settlements"]:
                                pending = nid; break
                        if pending is not None:
                            u,v = e
                            if u == pending or v == pending:
                                self.setup_road_assigned[pi].add(pending)
                else:
                    self.toast("Illegal: Road placement not legal", "error")
            elif data in ("settlement","city") and target[0] == "node":
                nid = target[1]
                if data == "settlement":
                    if self.is_legal_settlement(pi, nid):
                        self.players[pi]["cities"].discard(nid)
                        self.players[pi]["settlements"].add(nid)
                        # Track order always
                        if nid in self.setup_settlement_order[pi]:
                            self.setup_settlement_order[pi].remove(nid)
                        self.setup_settlement_order[pi].append(nid)
                        if len(self.setup_settlement_order[pi]) >= 2:
                            self.second_settlement[pi] = self.setup_settlement_order[pi][-1]
                    else:
                        self.toast("Illegal: Settlement placement not legal", "error")
                else:
                    if self.is_legal_city(pi, nid):
                        self.players[pi]["settlements"].discard(nid)
                        self.players[pi]["cities"].add(nid)
                    else:
                        self.toast("Illegal: City must upgrade a settlement", "error")
        elif kind == "piece_erase":
            typ, val = target
            if typ == "node":
                self.players[pi]["settlements"].discard(val)
                self.players[pi]["cities"].discard(val)
                if val in self.setup_settlement_order[pi]:
                    self.setup_settlement_order[pi].remove(val)
                self.setup_road_assigned[pi].discard(val)
                # Recompute second settlement if needed
                if self.second_settlement.get(pi) == val:
                    if len(self.setup_settlement_order[pi]) >= 2:
                        self.second_settlement[pi] = self.setup_settlement_order[pi][-1]
                    else:
                        self.second_settlement.pop(pi, None)
            elif typ == "edge":
                # Do not allow removing the only road attached to a settlement (keeps board consistent)
                if self.enforce_rules.get():
                    u,v = val
                    for nid in (u, v):
                        if nid in self.players[pi]["settlements"]:
                            # any other road touching this nid (besides val)?
                            if not any(((a==nid or b==nid) and (a,b)!=val) for (a,b) in self.players[pi]["roads"]):
                                self.toast("Setup: cannot remove the only road attached to a settlement", "error")
                                self.redraw(); return
                self.players[pi]["roads"].discard(val)
                if self.enforce_rules.get() and self.setup_mode.get():
                    u,v = val
                    for nid in (u, v):
                        if nid in self.setup_road_assigned[pi] and not self.has_adjacent_road(pi, nid):
                            self.setup_road_assigned[pi].discard(nid)
        elif kind == "resource" and target[0] == "hex":
            hid = target[1]
            old = self.hex_resources.get(hid); new = data
            if self.enforce_rules.get() and new is not UNSET:
                cur = Counter(v for v in self.hex_resources.values() if v is not UNSET)
                used = cur.get(new, 0) - (1 if old == new else 0)
                limit = RESOURCE_LIMITS.get(new, None)
                if limit is not None and used + 1 > limit:
                    self.toast(f"Limit reached: too many {new if new is not None else 'desert'} tiles", "warn")
                    self.redraw(); return
            self.hex_resources[hid] = new
            if new is None:
                self.hex_numbers[hid] = None
        elif kind == "number" and target[0] == "hex":
            hid = target[1]
            rv = self.hex_resources.get(hid)
            if rv is None or rv is UNSET:
                self.toast("Illegal: number on empty/desert hex", "error")
            else:
                val = int(data)
                if self.enforce_rules.get():
                    cur = Counter(v for v in self.hex_numbers.values() if v is not None)
                    if self.hex_numbers.get(hid) != val and cur.get(val, 0) >= TOKEN_LIMITS[val]:
                        self.toast(f"Limit reached: all {val} tokens used", "warn")
                        self.redraw(); return
                self.hex_numbers[hid] = val
        elif kind == "robber" and target[0] == "hex":
            self.robber_hex = target[1]
        elif kind == "port" and target[0] == "port":
            nid = target[1]
            if not self.is_border_node(nid):
                self.toast("Illegal: ports only on border nodes", "error")
            else:
                new_type = data.get("type")
                new_key = "generic" if new_type=="generic" else data.get("resource")
                old = self.board.ports.get(nid)
                # compute effective used before placing
                port_counts = Counter()
                for n,p in self.board.ports.items():
                    if p.get("type")=="generic": port_counts["generic"] += 1
                    elif p.get("type")=="resource": port_counts[p.get("resource")] += 1
                if old is not None:
                    if old.get("type")=="generic": port_counts["generic"] -= 1
                    elif old.get("type")=="resource": port_counts[old.get("resource")] -= 1
                if self.enforce_rules.get() and port_counts.get(new_key, 0) >= PORT_LIMITS[new_key]:
                    msg = "3:1" if new_key=="generic" else f"2:1 {new_key}"
                    self.toast(f"Limit reached: all {msg} ports placed", "warn")
                else:
                    self.board.ports[nid] = {"type": data["type"], "resource": data.get("resource"), "rate": data["rate"]}
        elif kind == "port_erase" and target[0] == "port":
            nid = target[1]
            if nid in self.board.ports:
                del self.board.ports[nid]
        self.redraw()

    def on_global_drag(self, event):
        if self.dragging is None:
            return
        cx = self.canvas.winfo_rootx(); cy = self.canvas.winfo_rooty()
        x = event.x_root - cx; y = event.y_root - cy
        if 0 <= x <= self.canvas.winfo_width() and 0 <= y <= self.canvas.winfo_height():
            self.dragging["pos"] = (x, y)
        else:
            self.dragging["pos"] = None
            self.dragging["target"] = None
        self.redraw()

    def on_global_drop(self, event):
        if self.dragging is None:
            return
        cx = self.canvas.winfo_rootx(); cy = self.canvas.winfo_rooty()
        x = event.x_root - cx; y = event.y_root - cy
        if 0 <= x <= self.canvas.winfo_width() and 0 <= y <= self.canvas.winfo_height():
            self.dragging["pos"] = (x, y)
            self.redraw()
            self.on_canvas_drop(event)
        else:
            self.dragging = None

    # ---------- Actions & Simulation ----------
    def toast(self, msg: str, kind: str="info"):
        bg = {"info":"#24395e","error":"#5b2b2b","warn":"#664d1a"}.get(kind, "#24395e")
        fg = "#e6eefb"
        lbl = tk.Label(self.root, text=msg, bg=bg, fg=fg, padx=10, pady=4)
        lbl.place(relx=0.98, rely=0.02, anchor="ne")
        self.root.after(1800, lbl.destroy)
    def clear_pieces(self):
        for i in range(4):
            self.players[i]["settlements"].clear()
            self.players[i]["cities"].clear()
            self.players[i]["roads"].clear()
        self.redraw()

    def randomize_board(self):
        pool = ([None]*1 + ["brick"]*3 + ["wood"]*4 + ["wool"]*4 + ["grain"]*4 + ["ore"]*3)
        import random
        random.shuffle(pool)
        for hid in self.board.hexes.keys():
            self.hex_resources[hid] = pool.pop()
            self.hex_numbers[hid] = None
        self.robber_hex = next((hid for hid,r in self.hex_resources.items() if r is None), 0)
        self.redraw()

    def reset_board(self):
        self.board = make_standard_board(seed=0)
        self.hex_resources = {hid: UNSET for hid in self.board.hexes.keys()}
        self.hex_numbers = {hid: None for hid in self.board.hexes.keys()}
        self.board.ports = {}
        self.robber_hex = 0
        self.clear_pieces()

    def _apply_init_state(self, data: dict):
        """Apply an init dict with keys {board, players} into the GUI/editor state."""
        try:
            b = data.get("board", {})
            hexes = b.get("hexes", {})
            for hid_str, h in hexes.items():
                hid = int(hid_str)
                res = h.get("resource", None)
                # In editor, UNSET means empty tile (no resource/number), distinct from desert None
                self.hex_resources[hid] = UNSET if res is None and h.get("number") is None else res
                self.hex_numbers[hid] = h.get("number", None)
            # Ports and robber
            self.board.ports = {int(k): v for k, v in b.get("ports", {}).items()}
            self.robber_hex = int(b.get("robber_hex", 0))

            # Players and pieces
            players = data.get("players", [])
            N = max(2, min(4, len(players))) if players else self.num_players.get()
            self.num_players.set(N)
            for i in range(4):
                self.player_names[i].set(f"Player {i+1}")
                self.players[i]["settlements"].clear()
                self.players[i]["cities"].clear()
                self.players[i]["roads"].clear()
                self.setup_settlement_order[i].clear()
                self.setup_road_assigned[i].clear()
            for i, p in enumerate(players[:4]):
                self.player_names[i].set(p.get("name", f"Player {i+1}"))
                # Normalize sets
                for n in p.get("settlements", []):
                    try:
                        self.players[i]["settlements"].add(int(n))
                    except Exception:
                        pass
                for n in p.get("cities", []):
                    try:
                        self.players[i]["cities"].add(int(n))
                    except Exception:
                        pass
                for e in p.get("roads", []):
                    try:
                        u, v = int(e[0]), int(e[1])
                        self.players[i]["roads"].add(tuple(sorted((u, v))))
                    except Exception:
                        pass

            # Derive setup meta: order and road assignment if it looks like setup
            looks_like_setup = True
            for i in range(self.num_players.get()):
                if len(self.players[i]["cities"]) != 0:
                    looks_like_setup = False
                if len(self.players[i]["settlements"]) > 2 or len(self.players[i]["roads"]) > 2:
                    looks_like_setup = False
            # Populate order (arbitrary but stable) and assigned-road flags
            for i in range(self.num_players.get()):
                ordered = list(sorted(self.players[i]["settlements"]))
                self.setup_settlement_order[i] = ordered.copy()
                assigned = set()
                for nid in ordered:
                    if self.has_adjacent_road(i, nid):
                        assigned.add(nid)
                self.setup_road_assigned[i] = assigned
                if len(ordered) >= 2:
                    self.second_settlement[i] = ordered[-1]
                elif i in self.second_settlement:
                    self.second_settlement.pop(i, None)

            if looks_like_setup:
                self.setup_mode.set(True)
            self.redraw(); self.update_counters(); self.update_setup_progress()
        except Exception as e:
            self.toast(str(e), "error")

    def randomize_full_board(self):
        """Randomly assign resources and number tokens, and set robber on desert.
        This uses standard counts (1 desert, 3 brick, 4 wood, 4 wool, 4 grain, 3 ore)
        and the standard 18 number tokens in random positions on non-desert tiles.
        """
        # Randomize resources first
        self.randomize_board()
        # Assign tokens randomly to non-desert tiles
        import random
        non_desert_hids = [hid for hid in self.board.hexes.keys() if self.hex_resources.get(hid) not in (None, UNSET)]
        random.shuffle(non_desert_hids)
        tokens = list(TOKEN_ORDER)
        for i, hid in enumerate(non_desert_hids[:len(tokens)]):
            self.hex_numbers[hid] = tokens[i]
        # Ensure robber is on a desert (if any)
        desert_hids = [hid for hid in self.board.hexes.keys() if self.hex_resources.get(hid) is None]
        if desert_hids:
            self.robber_hex = desert_hids[0]
        # Sync editor state into engine board before ports/auto-place
        self.sync_editor_to_board()
        # Randomize ports on border nodes
        self.randomize_ports()
        # Auto-place players' initial settlements and roads using bots
        self.auto_place_by_bots(use_strong=self.use_strong_bots.get())
        self.redraw()

    def sync_editor_to_board(self):
        """Write current editor resources/numbers/robber into the engine Board so
        bots and simulations operate on what is visible in the GUI."""
        for hid, h in self.board.hexes.items():
            rv = self.hex_resources.get(hid)
            h.resource = None if rv is UNSET else rv
            h.number = self.hex_numbers.get(hid)
        self.board.robber_hex = int(self.robber_hex)

    def randomize_ports(self):
        """Place 9 ports in a fixed, official-like order around the perimeter.
        Approach:
        - Collect border nodes (deg<=2)
        - Sort by angle around centroid (clockwise ring)
        - Pick 9 nodes by equal step to distribute evenly
        - Assign the canonical port pattern clockwise
        Pattern used (clockwise):
            [3:1, ore, 3:1, grain, 3:1, wool, 3:1, brick, wood]
        This matches standard counts (4 generic, 5 specific) and a common base-game layout.
        """
        import math
        nodes = self.board.nodes
        border = [nid for nid in nodes.keys() if len(self.board.adj.get(nid, ())) <= 2]
        if not border:
            self.board.ports = {}
            return
        cx = sum(nodes[n][0] for n in border) / len(border)
        cy = sum(nodes[n][1] for n in border) / len(border)
        # Sort clockwise starting from the topmost (smallest y) to stabilize
        border.sort(key=lambda n: math.atan2(nodes[n][1]-cy, nodes[n][0]-cx))
        step = max(1, int(round(len(border)/9)))
        chosen = []
        used = set(); idx = 0
        while len(chosen) < 9 and border:
            n = border[idx % len(border)]
            if n not in used:
                chosen.append(n); used.add(n)
            idx += step
            if idx > len(border)*3:
                break
        # Fallback fill
        i = 0
        while len(chosen) < 9 and i < len(border):
            n = border[i]
            if n not in used:
                chosen.append(n); used.add(n)
            i += 1
        # Fixed canonical pattern
        pattern = [
            {"type":"generic","resource":None,"rate":3},
            {"type":"resource","resource":"ore","rate":2},
            {"type":"generic","resource":None,"rate":3},
            {"type":"resource","resource":"grain","rate":2},
            {"type":"generic","resource":None,"rate":3},
            {"type":"resource","resource":"wool","rate":2},
            {"type":"generic","resource":None,"rate":3},
            {"type":"resource","resource":"brick","rate":2},
            {"type":"resource","resource":"wood","rate":2},
        ]
        ports = {}
        for nid, spec in zip(chosen, pattern):
            ports[nid] = spec
        self.board.ports = ports

    def auto_place_by_bots(self, use_strong: bool=True):
        """Use engine bots to place setup settlements/roads in snake order, and animate
        the placement step-by-step to reflect Catan rules (one by one)."""
        try:
            import random
            N = self.num_players.get()
            g = Game(seed=random.randint(0, 10_000), target_vp=10, max_turns=1,
                     use_mcts_bot=use_strong, num_players=N, mcts_all=use_strong,
                     mcts_beam_width=int(self.mcts_beam_var.get()),
                     mcts_max_depth=int(self.mcts_depth_var.get()))
            # override board with our current one (resources/numbers/ports already set)
            g.board = self.board
            for i in range(N):
                g.players[i].name = self.player_names[i].get() or f"Player {i+1}"
            # Run placements; snapshots are recorded per step
            g.setup_initial_placements()

            # Helper to apply a trace frame to the GUI state
            def apply_frame(frame, prev_sets=None, step_index=None, players_count=None):
                fps = frame.get('players', [])
                for i in range(4):
                    self.players[i]["settlements"].clear(); self.players[i]["cities"].clear(); self.players[i]["roads"].clear()
                for p in fps:
                    pid = int(p.get('id', 0))
                    if pid < 0 or pid >= 4:
                        continue
                    self.players[pid]["settlements"].update(int(n) for n in p.get('settlements', []))
                    self.players[pid]["cities"].update(int(n) for n in p.get('cities', []))
                    self.players[pid]["roads"].update(tuple(sorted((int(e[0]), int(e[1])))) for e in p.get('roads', []))
                # Mark setup tracking from this frame
                # If prev_sets provided, detect which player added a settlement this step and record order
                if prev_sets is not None and step_index is not None and players_count is not None:
                    cur_sets = {int(p.get('id',0)): set(int(n) for n in p.get('settlements', [])) for p in fps}
                    for pid, cur in cur_sets.items():
                        before = prev_sets.get(pid, set())
                        added = list(cur - before)
                        if added:
                            nid = added[0]
                            if nid in self.setup_settlement_order[pid]:
                                self.setup_settlement_order[pid].remove(nid)
                            self.setup_settlement_order[pid].append(nid)
                            if step_index >= players_count:
                                self.second_settlement[pid] = nid
                    prev_sets.clear(); prev_sets.update(cur_sets)
                # Assume any current settlements have roads assigned in setup visuals
                for i in range(N):
                    self.setup_road_assigned[i] = set(self.players[i]["settlements"]) 
                self.redraw(); self.update_counters(); self.update_setup_progress(); self.root.update_idletasks()

            # Animate only the initial placement frames
            steps = [f for f in g.trace if isinstance(f.get('note',''), str) and f['note'].startswith('Initial placement')]
            if not steps:
                # Fallback: apply final state
                apply_frame(g.trace[-1] if g.trace else {})
                self.toast("Auto-placed initial settlements and roads")
                return

            prev_sets = {p.id: set() for p in g.players}
            def animate(i=0):
                if i >= len(steps):
                    self.toast("Auto-placed initial settlements and roads")
                    return
                apply_frame(steps[i], prev_sets=prev_sets, step_index=i, players_count=N)
                # brief pause to show sequential placement
                self.root.after(250, lambda: animate(i+1))

            animate(0)
        except Exception as e:
            self.toast(f"Auto-place failed: {e}", "error")
        
    # ---------- Replay: simulate and step through full game ----------
    def capture_game_to_replay(self):
        # Build init from current editor state and run a single game, then load its trace
        try:
            turns = int(self.turns_var.get().strip())
        except Exception:
            self.toast("Invalid input: Max Turns", "error"); return
        init = self.build_init_json()
        self.status.config(text="Capturing game trace...")
        self.capture_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.DISABLED)
        self.run_one_btn.config(state=tk.DISABLED)

        def worker():
            try:
                import random
                seed = random.randint(0, 1_000_000)
                g = Game(seed=seed,
                         target_vp=int(init.get("target_vp", 10)),
                         max_turns=turns,
                         use_mcts_bot=self.use_strong_bots.get(),
                         num_players=len(init["players"]),
                         mcts_all=self.use_strong_bots.get(),
                         mcts_beam_width=int(self.mcts_beam_var.get()),
                         mcts_max_depth=int(self.mcts_depth_var.get()))
                from starter_eval import build_board, build_players
                g.board = build_board(init["board"])  # use GUI board
                g.players = build_players(init["players"])  # placements and starting hands
                from catan_plus import HeuristicBot, MCTSBot
                g.bots = {p.id: (MCTSBot(p.id, p.name) if (self.use_strong_bots.get() and (g.mcts_all or p.id==0)) else HeuristicBot(p.id, p.name)) for p in g.players}
                # Start and play
                g.snapshot("Start from GUI state")
                g.play_loop()
                trace = g.trace[:]  # list of frames
                logs = g.logs[:]
                # Apply on UI thread
                self.root.after(0, lambda: self._replay_set_data(trace, logs))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda m=msg: (self.toast(m, "error"), self.status.config(text="Error")))
            finally:
                self.root.after(0, lambda: (self.capture_btn.config(state=tk.NORMAL), self.run_btn.config(state=tk.NORMAL), self.run_one_btn.config(state=tk.NORMAL)))

        threading.Thread(target=worker, daemon=True).start()

    def _replay_set_data(self, trace: list[dict], logs: list[str]):
        self.replay_trace = trace or []
        self.replay_logs = logs or []
        n = len(self.replay_trace)
        self.replay_scale.config(to=max(0, n-1))
        self.replay_index.set(0 if n else 0)
        self.replay_status.config(text=f"0/{n}")
        # Show first frame immediately
        if n:
            self.replay_goto(0)
            self.status.config(text=f"Trace ready ({n} steps)")
        else:
            self.status.config(text="No steps captured")

    def replay_goto(self, idx: int):
        if not self.replay_trace:
            return
        idx = max(0, min(int(idx), len(self.replay_trace)-1))
        self.replay_index.set(idx)
        self.replay_status.config(text=f"{idx+1}/{len(self.replay_trace)}")
        frame = self.replay_trace[idx]
        self._apply_trace_frame(frame)

    def replay_first(self):
        self.replay_goto(0)

    def replay_last(self):
        if self.replay_trace:
            self.replay_goto(len(self.replay_trace)-1)

    def replay_prev(self):
        self.replay_goto(self.replay_index.get()-1)

    def replay_next(self):
        self.replay_goto(self.replay_index.get()+1)
        if self.replay_playing and self.replay_index.get() < max(0, len(self.replay_trace)-1):
            self.root.after(350, self.replay_next)
        else:
            if self.replay_playing:
                self.replay_playing = False
                self.play_btn.config(text="Play")

    def replay_toggle_play(self):
        if not self.replay_trace:
            return
        self.replay_playing = not self.replay_playing
        self.play_btn.config(text=("Pause" if self.replay_playing else "Play"))
        if self.replay_playing:
            # If at end, restart
            if self.replay_index.get() >= len(self.replay_trace)-1:
                self.replay_first()
            self.root.after(200, self.replay_next)

    def _apply_trace_frame(self, frame: dict):
        # Sync board attributes from frame
        try:
            b = frame.get('board', {})
            # Update hex resources/numbers if present
            hexes = b.get('hexes') or {}
            if hexes:
                for hid_str, h in hexes.items():
                    hid = int(hid_str) if isinstance(hid_str, str) else hid_str
                    rv = h.get('resource', None)
                    self.hex_resources[hid] = (UNSET if rv is None and 'resource' not in h else rv)
                    self.hex_numbers[hid] = h.get('number', None)
            # Robber and ports
            if 'robber_hex' in b:
                self.robber_hex = int(b.get('robber_hex', self.robber_hex))
            if 'ports' in b and isinstance(b['ports'], dict):
                self.board.ports = b['ports']
        except Exception:
            pass

        # Apply players' pieces
        for i in range(4):
            self.players[i]["settlements"].clear(); self.players[i]["cities"].clear(); self.players[i]["roads"].clear()
        fps = frame.get('players', [])
        for p in fps:
            try:
                pid = int(p.get('id', 0))
                if 0 <= pid < 4:
                    self.players[pid]["settlements"].update(int(n) for n in p.get('settlements', []))
                    self.players[pid]["cities"].update(int(n) for n in p.get('cities', []))
                    self.players[pid]["roads"].update(tuple(sorted((int(e[0]), int(e[1])))) for e in p.get('roads', []))
            except Exception:
                continue

        # Update textual details and HUD datapoints
        note = frame.get('note', '') or ''
        turn = frame.get('turn', 0)
        cur = frame.get('current', 0)
        self.replay_note.config(text=f"T{turn} P{cur+1}: {note}")
        self._update_replay_info(frame)
        # Prepare HUD player list in index order 0..N-1
        fps = frame.get('players', [])
        max_pid = -1
        tmp = {}
        for p in fps:
            try:
                pid = int(p.get('id', 0)); max_pid = max(max_pid, pid)
                tmp[pid] = {
                    'vp': p.get('vp', 0),
                    'hidden_vp': p.get('hidden_vp', 0),
                    'hand': p.get('hand', {}) or {},
                }
            except Exception:
                pass
        N = max(self.num_players.get(), max_pid+1 if max_pid>=0 else 0)
        self.hud_players = [tmp.get(i, {'vp':0,'hidden_vp':0,'hand':{}}) for i in range(N)]
        # Extract dice roll if present in note
        roll = None
        if isinstance(note, str):
            if note.startswith('Production '):
                try:
                    roll = int(note.split(' ', 1)[1])
                except Exception:
                    roll = None
            elif 'robber' in note.lower():
                roll = 7
        self.hud_roll = roll
        self.hud_current = cur if isinstance(cur, int) else None
        self.redraw(); self.update_idletasks_safe()

    def _update_replay_info(self, frame: dict):
        self.replay_info.delete('1.0', tk.END)
        fps = frame.get('players', [])
        lines = []
        for p in fps:
            pid = int(p.get('id', 0))
            name = p.get('name', f'P{pid+1}')
            vp = p.get('vp', 0); hvp = p.get('hidden_vp', 0)
            hand = p.get('hand', {}) or {}
            dev = p.get('dev_hand', {}) or {}
            lr = '✓' if p.get('has_longest_road') else ' '
            la = '✓' if p.get('has_largest_army') else ' '
            hand_txt = ', '.join(f"{k}:{v}" for k,v in sorted(hand.items())) or '-'
            dev_txt = ', '.join(f"{k}:{v}" for k,v in sorted(dev.items())) or '-'
            lines.append(f"P{pid+1} {name} | VP {vp}+{hvp}* | Hand {hand_txt} | Dev {dev_txt} | LR {lr} LA {la}")
        self.replay_info.insert(tk.END, "\n".join(lines))

    def update_idletasks_safe(self):
        try:
            self.root.update_idletasks()
        except Exception:
            pass

    def on_zoom(self, event):
        try:
            delta = event.delta
        except Exception:
            delta = 0
        factor = 1.1 if delta > 0 else 0.9
        self.scale = max(0.5, min(2.0, self.scale * factor))
        self.redraw()

    def on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        if not self._pan_start:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self.pan[0] += dx
        self.pan[1] += dy
        self.redraw()

    def save_board(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        data = self.build_init_json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.toast("Board saved")

    def load_board(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_init_state(data)
            self.toast("Board loaded")
        except Exception as e:
            self.toast(str(e), "error")

    def import_from_image(self):
        """Pick an image and best-effort parse a board+setup from it.
        Prefers QR/state embedded or sibling trace.json; falls back to a heuristic visual parser.
        """
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"),
            ("All Files", "*.*"),
        ])
        if not path:
            return
        # Run heavy parsing off the UI thread to avoid freezing
        def worker(image_path: str):
            try:
                from image_eval import load_state_from_qr_only, extract_state_from_visualizer_image
            except Exception:
                def fail():
                    self.toast("image_eval not available", "error")
                    self.run_btn.config(state=tk.NORMAL)
                    self.run_one_btn.config(state=tk.NORMAL)
                    self.status.config(text="Idle")
                self.root.after(0, fail)
                return
            init = None
            try:
                init = load_state_from_qr_only(image_path)
            except Exception:
                init = None
            if init is None:
                try:
                    # Provide cancel/progress hooks for responsive UI
                    def is_canceled():
                        return bool(getattr(self, '_import_cancel', None) and self._import_cancel.is_set())
                    def prog(pct: float, msg: str):
                        dlg = getattr(self, '_import_dialog', None)
                        if not dlg:
                            return
                        try:
                            self._import_prog['value'] = max(0, min(100, pct))
                            self._import_msg_var.set(msg)
                        except Exception:
                            pass
                    parsed = extract_state_from_visualizer_image(image_path, cancel_check=is_canceled, progress=lambda p,m: self.root.after(0, prog, p, m))
                except Exception:
                    parsed = None
                if isinstance(parsed, tuple):
                    init = parsed[0]
                else:
                    init = parsed
            # Apply on UI thread
            def apply_result():
                self.run_btn.config(state=tk.NORMAL)
                self.run_one_btn.config(state=tk.NORMAL)
                self.status.config(text="Idle")
                # Close dialog if open
                try:
                    if getattr(self, '_import_dialog', None):
                        self._import_dialog.destroy()
                        self._import_dialog = None
                except Exception:
                    pass
                if not isinstance(init, dict) or "board" not in init or "players" not in init:
                    if getattr(self, '_import_cancel', None) and self._import_cancel.is_set():
                        self.toast("Import canceled", "warn")
                    else:
                        self.toast("Could not parse image", "error")
                    return
                self._apply_init_state(init)
                self.toast("Loaded from image")
            self.root.after(0, apply_result)

        # Disable actions and show progress dialog, then start worker
        self.status.config(text="Parsing image...")
        self.run_btn.config(state=tk.DISABLED)
        self.run_one_btn.config(state=tk.DISABLED)
        # Progress dialog
        try:
            dlg = tk.Toplevel(self.root)
            dlg.title("Importing from Image")
            dlg.transient(self.root)
            dlg.resizable(False, False)
            ttk.Label(dlg, text="Please wait while we parse the image...").pack(padx=12, pady=(10,6))
            self._import_msg_var = tk.StringVar(value="Starting...")
            ttk.Label(dlg, textvariable=self._import_msg_var).pack(padx=12, pady=(0,6))
            self._import_prog = ttk.Progressbar(dlg, mode='determinate', length=360)
            self._import_prog.pack(padx=12, pady=(0,10))
            btns = ttk.Frame(dlg); btns.pack(pady=(0,10))
            def on_cancel():
                if getattr(self, '_import_cancel', None):
                    self._import_cancel.set()
                self._import_msg_var.set("Canceling...")
            ttk.Button(btns, text="Cancel", command=on_cancel).pack()
            dlg.update_idletasks()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - dlg.winfo_width())//2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_height())//2
            dlg.geometry(f"+{x}+{y}")
            self._import_dialog = dlg
        except Exception:
            self._import_dialog = None
        # Cancel flag
        import threading as _th
        self._import_cancel = _th.Event()
        threading.Thread(target=worker, args=(path,), daemon=True).start()

    def build_init_json(self):
        # Ensure engine board hex info is in sync with editor before creating init
        self.sync_editor_to_board()
        nodes = {str(k): list(v) for k,v in self.board.nodes.items()}
        hexes = {}
        for hid, h in self.board.hexes.items():
            rv = self.hex_resources.get(hid)
            hexes[str(hid)] = {"resource": (None if rv is UNSET else rv), "number": self.hex_numbers.get(hid), "nodes": h.nodes}
        edges = [list(e) for e in self.board.edges]
        ports = self.board.ports
        board_json = {"nodes": nodes, "hexes": hexes, "edges": edges, "ports": ports, "robber_hex": self.robber_hex}
        N = self.num_players.get()
        players = []
        for i in range(N):
            # Compute starting hand from second settlement (official starting resources)
            hand = Counter()
            second = self.second_settlement.get(i)
            if second is None and len(self.setup_settlement_order[i]) >= 2:
                second = self.setup_settlement_order[i][-1]
            if second is not None:
                for hid in self.board.node_to_hexes.get(second, []):
                    h = self.board.hexes[hid]
                    if h.resource:
                        hand[h.resource] += 1
            players.append({
                "name": self.player_names[i].get() or f"Player {i+1}",
                "settlements": sorted(self.players[i]["settlements"]),
                "cities": sorted(self.players[i]["cities"]),
                "roads": [list(e) for e in sorted(self.players[i]["roads"])],
                "hand": dict(hand),
            })
        return {"seed": 0, "target_vp": 10, "board": board_json, "players": players}

    def run_simulation(self):
        try:
            games = int(self.games_var.get().strip())
            turns = int(self.turns_var.get().strip())
        except Exception:
            self.toast("Invalid input: #Games and Max Turns", "error")
            return
        init = self.build_init_json()
        self.status.config(text=f"Simulating {games} games...")
        self.run_btn.config(state=tk.DISABLED)
        self.run_one_btn.config(state=tk.DISABLED)
        self.results_txt.delete("1.0", tk.END)
        def worker():
            try:
                wins, turns_list = simulate(
                    init,
                    games=games,
                    max_turns=turns,
                    use_mcts=self.use_strong_bots.get(),
                    mcts_all=self.use_strong_bots.get(),
                    mcts_beam_width=int(self.mcts_beam_var.get()),
                    mcts_max_depth=int(self.mcts_depth_var.get()),
                )
                total = sum(wins) if sum(wins) else 1
                names = [p['name'] for p in init['players']]
                results = {
                    'players': [
                        {'id': i, 'name': names[i], 'wins': wins[i], 'win_rate': wins[i]/total}
                        for i in range(len(names))
                    ],
                    'games': games,
                    'avg_turns': (sum(turns_list)/len(turns_list)) if turns_list else None,
                }
                self.root.after(0, lambda: self.results_txt.insert(tk.END, json.dumps(results, indent=2)))
                self.root.after(0, lambda: self.status.config(text="Done"))
            except Exception as e:
                self.root.after(0, lambda: self.toast(str(e), "error"))
                self.root.after(0, lambda: self.status.config(text="Error"))
            finally:
                self.root.after(0, lambda: (self.run_btn.config(state=tk.NORMAL), self.run_one_btn.config(state=tk.NORMAL)))
        threading.Thread(target=worker, daemon=True).start()

    def run_single_game(self):
        # Run exactly one game and then update board to final state and print logs
        try:
            turns = int(self.turns_var.get().strip())
        except Exception:
            self.toast("Invalid input: Max Turns", "error")
            return
        init = self.build_init_json()
        self.status.config(text="Simulating 1 game...")
        self.run_btn.config(state=tk.DISABLED)
        self.run_one_btn.config(state=tk.DISABLED)
        self.results_txt.delete("1.0", tk.END)

        def worker():
            try:
                # Build game from current GUI state
                import random
                seed = random.randint(0, 1_000_000)
                g = Game(seed=seed,
                        target_vp=int(init.get("target_vp", 10)),
                        max_turns=turns,
                        use_mcts_bot=self.use_strong_bots.get(),
                        num_players=len(init["players"]),
                        mcts_all=self.use_strong_bots.get(),
                        mcts_beam_width=int(self.mcts_beam_var.get()),
                        mcts_max_depth=int(self.mcts_depth_var.get()))
                g.board = build_board(init["board"])  # use GUI board
                g.players = build_players(init["players"])  # placements and starting hands
                # Rebuild bots to align with players
                from catan_plus import HeuristicBot, MCTSBot
                g.bots = {p.id: (MCTSBot(p.id, p.name) if (self.use_strong_bots.get() and (g.mcts_all or p.id==0)) else HeuristicBot(p.id, p.name)) for p in g.players}
                # Start and play
                g.snapshot("Start from GUI state")
                g.play_loop()
                # Determine winner (mirror starter_eval logic)
                best = max(range(len(g.players)), key=lambda i: (g.players[i].vp + g.players[i].hidden_vp, sum(g.players[i].hand.values()), -i))
                winner = g.players[best]

                # Build a text log
                summary = {
                    'winner': {'id': winner.id, 'name': winner.name, 'vp': winner.vp, 'hidden_vp': winner.hidden_vp},
                    'turns': g.turn,
                    'largest_army_holder': g.largest_army_holder,
                    'longest_road_holder': g.longest_road_holder,
                    'seed': seed,
                }
                text = "".join(line + "\n" for line in g.logs)
                out = json.dumps(summary, indent=2) + "\n\n" + text

                # Push GUI updates on main thread
                def apply_final_state():
                    # Update player pieces in GUI
                    N = self.num_players.get()
                    for i in range(N):
                        self.players[i]["settlements"].clear()
                        self.players[i]["cities"].clear()
                        self.players[i]["roads"].clear()
                    for p in g.players:
                        pid = p.id
                        if pid < 4:
                            self.players[pid]["settlements"] = set(int(n) for n in p.settlements)
                            self.players[pid]["cities"] = set(int(n) for n in p.cities)
                            self.players[pid]["roads"] = set(tuple(int(x) for x in e) for e in p.roads)
                    # Robber position and redraw
                    self.robber_hex = g.board.robber_hex
                    self.redraw()
                    self.results_txt.insert(tk.END, out)
                    self.status.config(text=f"Done: {winner.name} wins in {g.turn} turns")

                self.root.after(0, apply_final_state)
            except Exception as e:
                # Exception variables are cleared after the except block; capture message for callback
                msg = str(e)
                self.root.after(0, lambda m=msg: (self.toast(m, "error"), self.status.config(text="Error")))
            finally:
                self.root.after(0, lambda: (self.run_btn.config(state=tk.NORMAL), self.run_one_btn.config(state=tk.NORMAL)))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Counters ----------
    def update_counters(self):
        cur = Counter(v for v in self.hex_resources.values() if v is not UNSET)
        for r,lbl in self.resource_count_labels.items():
            limit = RESOURCE_LIMITS.get(r, 0); used = cur.get(r, 0)
            rem = max(0, limit - used)
            lbl.config(text=str(rem), relief=(tk.SUNKEN if rem==0 else tk.RIDGE))
        curN = Counter(v for v in self.hex_numbers.values() if v is not None)
        for n,lbl in self.number_count_labels.items():
            limit = TOKEN_LIMITS[n]; used = curN.get(n, 0)
            rem = max(0, limit - used)
            lbl.config(text=str(rem), relief=(tk.SUNKEN if rem==0 else tk.RIDGE))
        # Ports remaining
        port_counts = Counter()
        for p in self.board.ports.values():
            if p.get("type") == "generic":
                port_counts["generic"] += 1
            elif p.get("type") == "resource":
                port_counts[p.get("resource")] += 1
        for key,lbl in self.port_count_labels.items():
            limit = PORT_LIMITS[key]
            used = port_counts.get(key, 0)
            rem = max(0, limit - used)
            lbl.config(text=str(rem), relief=(tk.SUNKEN if rem==0 else tk.RIDGE))

    def update_setup_progress(self):
        # Show settlement count and which settlements still need roads (in setup mode)
        N = self.num_players.get()
        for i, lbl in enumerate(self.setup_progress_labels):
            if i >= N:
                lbl.config(text="")
                continue
            s_count = len(self.players[i]["settlements"]) if self.players[i]["settlements"] else 0
            if self.setup_mode.get():
                pending = [nid for nid in self.setup_settlement_order[i]
                           if nid in self.players[i]["settlements"] and nid not in self.setup_road_assigned[i]]
                lbl.config(text=f"P{i+1}: Settlements {s_count}/2 | Roads pending: {pending}")
            else:
                r_count = len(self.players[i]["roads"]) if self.players[i]["roads"] else 0
                lbl.config(text=f"P{i+1}: Settlements {s_count} | Roads {r_count}")

def main():
    root = tk.Tk()
    # Apply optional ttkbootstrap modern theme if available
    try:
        if Style is not None:
            Style(theme="darkly")
    except Exception:
        pass
    app = BoardEditorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
