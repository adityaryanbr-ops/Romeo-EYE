import importlib.util
import os
import shutil
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
WEIGHTS_DIR = Path(os.environ.get("ROMEO_WEIGHTS_DIR", APP_DIR / "weights"))
DOWNLOADS = Path.home() / "Downloads"
STARTER = DOWNLOADS / "drone-detection-starter"
SNAP_DIR = APP_DIR / "snapshots"
REC_DIR = APP_DIR / "recordings"
CACHE_DIR = APP_DIR / ".model_cache"
BG_IMAGE = ASSETS_DIR / "romeo_bg.jpg"
ICON = ASSETS_DIR / "eye.ico"

# ---------- Models: first existing candidate wins ----------
MODEL_CHOICES = {
    "General": [
        WEIGHTS_DIR / "yolov8m.pt",
        DOWNLOADS / "yolov8m.pt",
        STARTER / "yolov8m.pt",
    ],
    "Drone Detection": [
        WEIGHTS_DIR / "drone.pt",
        STARTER / "runs/detect/birds_vs_drone_a100/weights/best.pt",
        DOWNLOADS / "detect/birds_vs_drone_a100/weights/best.pt",
    ],
    "OPERATIONAL DPLY": [
        WEIGHTS_DIR / "sard_humans.pt",
        STARTER / "runs/detect/sard_humans_a100/weights/best.pt",
    ],
    "DRONE VIEW": [
        WEIGHTS_DIR / "visdrone.pt",
        DOWNLOADS / "yolov8m-visdrone/best.pt",
        STARTER / "assets/visdrone/best.pt",
        STARTER / "weights/visdrone_best.pt",
    ],
    "WEAPON DETECTION": [
        WEIGHTS_DIR / "weapon.pt",
        STARTER / "assets/gunmodel/best3.pt",
        DOWNLOADS / "best3.pt",
    ],
}
CUSTOM_MODEL = "Custom weights (.pt)…"

SOURCE_CHOICES = ["Laptop Camera", "Local Video", "Image File", "IP Camera (HTTP/RTSP)"]
IMG_SIZES = ["320", "416", "480", "640", "960", "1280"]
THREAT_CLASSES = {"drone", "gun", "pistol", "rifle", "weapon", "knife", "handgun"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ---------- Theme ----------
BG = "#0a0e14"
PANEL = "#111823"
CARD = "#161f2e"
BORDER = "#1f2a3c"
FG = "#e6edf3"
MUTED = "#8b98a9"
ACCENT = "#22d3ee"
GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
FONT = "Segoe UI"


def resolve_weights(name):
    for p in MODEL_CHOICES.get(name, []):
        if Path(p).is_file():
            return str(p)
    return None


def pick_device():
    if torch is not None and torch.cuda.is_available():
        try:
            torch.ones(1, device="cuda").add_(1)
            return 0, True, torch.cuda.get_device_name(0)
        except Exception:
            pass
    return "cpu", False, "CPU"


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


class LatestFrameReader:
    """Reads a live source on its own thread and keeps only the newest frame,
    so inference never works on stale buffered frames."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.seq = 0
        self.alive = True
        self.cond = threading.Condition()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.alive:
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.alive = False
                    self.cond.notify_all()
                    return
                self.frame = frame
                self.seq += 1
                self.cond.notify_all()

    def read(self, last_seq, timeout=5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.alive, timeout)
            if self.seq == last_seq:
                return None, last_seq
            return self.frame, self.seq

    def stop(self):
        self.alive = False
        self.thread.join(timeout=1.0)


class ModelCache:
    def __init__(self, device, half):
        self.device = device
        self.half = half
        self.extra = {"half": True} if half else {}
        self._models = {}
        self._warm = set()
        self._lock = threading.Lock()

    def _openvino(self, path, notify):
        """On CPU, an OpenVINO export runs ~3x faster than PyTorch. Exported once, then reused."""
        try:
            import openvino  # noqa: F401
        except ImportError:
            return None
        src = Path(path)
        stat = src.stat()
        work = CACHE_DIR / f"{src.stem}_{stat.st_size:x}_{int(stat.st_mtime):x}"
        ov_dir = work / f"{src.stem}_openvino_model"
        try:
            if not (ov_dir / "metadata.yaml").exists():
                notify("Optimizing model for CPU (first run only)…")
                work.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, work / src.name)
                YOLO(str(work / src.name)).export(format="openvino", dynamic=True, half=False, verbose=False)
                (work / src.name).unlink(missing_ok=True)
            return YOLO(str(ov_dir), task="detect")
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            return None

    def get(self, path, imgsz, notify=lambda msg: None):
        with self._lock:
            model = self._models.get(path)
            if model is None:
                if self.device == "cpu":
                    model = self._openvino(path, notify)
                if model is None:
                    model = YOLO(path)
                self._models[path] = model
            key = (path, imgsz)
            if key not in self._warm:
                dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                model.predict(dummy, imgsz=imgsz, device=self.device, verbose=False, **self.extra)
                self._warm.add(key)
            return model


class HoverButton(tk.Button):
    def __init__(self, master, bg, hover, **kw):
        super().__init__(master, bg=bg, activebackground=hover, fg="white", activeforeground="white",
                         relief="flat", bd=0, cursor="hand2", font=(FONT, 11, "bold"),
                         padx=14, pady=9, disabledforeground="#5b6573", **kw)
        self._bg, self._hover = bg, hover
        self.bind("<Enter>", lambda e: self["state"] != "disabled" and self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))

    def set_colors(self, bg, hover):
        self._bg, self._hover = bg, hover
        self.config(bg=bg, activebackground=hover)


class RomeoEyeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROMEO EYE — Real-time Aerial Detection")
        self.root.configure(bg=BG)
        self.scale = max(self.root.winfo_fpixels("1i") / 96.0, 1.0)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(self.px(1360), sw - self.px(40))}x{min(self.px(820), sh - self.px(100))}+10+10")
        self.root.minsize(min(self.px(1000), sw), min(self.px(640), sh - self.px(100)))
        if sys.platform == "win32":
            self.root.state("zoomed")
        if ICON.exists():
            try:
                self.root.iconbitmap(str(ICON))
            except tk.TclError:
                pass

        self.device, self.half, self.device_name = pick_device()
        self.models = ModelCache(self.device, self.half)
        self.custom_weights = None

        self.running = False
        self.worker = None
        self.events = SimpleQueue()
        self._latest = None
        self._latest_lock = threading.Lock()
        self._last_annotated = None
        self._frame_tk = None
        self._placeholder_tk = None
        self._bg_src = Image.open(BG_IMAGE).convert("RGB") if BG_IMAGE.exists() else None
        self.display_size = (960, 600)
        self._last_stats_ui = 0.0
        self._fullscreen = False

        # Plain attributes read by the worker thread (Tk variables are not thread-safe).
        self.conf = 0.40
        self.imgsz = 640 if self.device != "cpu" else 416
        self.show_labels = True
        self.record = False

        self._setup_style()
        self._build_header()
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._build_sidebar(body)
        self._build_viewer(body)
        self._bind_keys()
        self._on_source_change()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(15, self._poll)

    def px(self, n):
        return int(n * self.scale)

    # ---------- UI construction ----------
    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=FG,
                        arrowcolor=ACCENT, bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD,
                        selectbackground=CARD, selectforeground=FG, padding=6)
        style.map("TCombobox",
                  fieldbackground=[("readonly", CARD), ("disabled", PANEL)],
                  foreground=[("disabled", MUTED)],
                  arrowcolor=[("disabled", MUTED)])
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=CARD,
                        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("TCheckbutton", background=PANEL, foreground=FG, font=(FONT, 10))
        style.map("TCheckbutton", background=[("active", PANEL)], indicatorcolor=[("selected", ACCENT)])
        self.root.option_add("*TCombobox*Listbox.background", CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", BG)
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 10))

    def _build_header(self):
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=20, pady=(14, 10))
        left = tk.Frame(head, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="◉ ROMEO EYE", font=(FONT, 24, "bold"), fg=ACCENT, bg=BG).pack(anchor="w")
        tk.Label(left, text="Real-time Aerial Detection Control Panel", font=(FONT, 10),
                 fg=MUTED, bg=BG).pack(anchor="w")

        right = tk.Frame(head, bg=BG)
        right.pack(side="right")
        self.threat_pill = tk.Label(right, text="● IDLE", font=(FONT, 10, "bold"), fg=MUTED, bg=CARD,
                                    padx=12, pady=6)
        self.threat_pill.pack(side="right", padx=(8, 0))
        dev_color = GREEN if self.device != "cpu" else AMBER
        dev_text = f"⚡ {self.device_name}" + (" · FP16" if self.half else "")
        if self.device == "cpu" and importlib.util.find_spec("openvino"):
            dev_text += " · OpenVINO"
        tk.Label(right, text=dev_text, font=(FONT, 10, "bold"), fg=dev_color, bg=CARD,
                 padx=12, pady=6).pack(side="right")

    def _section(self, parent, title):
        tk.Label(parent, text=title.upper(), font=(FONT, 9, "bold"), fg=MUTED, bg=PANEL).pack(
            anchor="w", padx=16, pady=(14, 4))

    def _build_sidebar(self, body):
        side = tk.Frame(body, bg=PANEL, width=self.px(310), highlightthickness=1, highlightbackground=BORDER)
        side.pack(side="left", fill="y", padx=(0, 14))
        side.pack_propagate(False)

        self._section(side, "Detection model")
        self.model_var = tk.StringVar(value=next(iter(MODEL_CHOICES)))
        self.model_box = ttk.Combobox(side, state="readonly", textvariable=self.model_var,
                                      values=list(MODEL_CHOICES) + [CUSTOM_MODEL], font=(FONT, 10))
        self.model_box.pack(fill="x", padx=16)
        self.model_box.bind("<<ComboboxSelected>>", self._on_model_change)
        self.model_info = tk.Label(side, text="", font=(FONT, 8), fg=MUTED, bg=PANEL, anchor="w",
                                   justify="left", wraplength=self.px(275))
        self.model_info.pack(fill="x", padx=16, pady=(4, 0))
        self._on_model_change()

        self._section(side, "Input source")
        self.source_var = tk.StringVar(value=SOURCE_CHOICES[0])
        self.source_box = ttk.Combobox(side, state="readonly", textvariable=self.source_var,
                                       values=SOURCE_CHOICES, font=(FONT, 10))
        self.source_box.pack(fill="x", padx=16)
        self.source_box.bind("<<ComboboxSelected>>", self._on_source_change)
        self.url_entry = tk.Entry(side, font=(FONT, 10), bg=CARD, fg=FG, insertbackground=FG,
                                  relief="flat", highlightthickness=1, highlightbackground=BORDER,
                                  highlightcolor=ACCENT, disabledbackground=PANEL,
                                  disabledforeground=MUTED)
        self.url_entry.insert(0, "rtsp://")
        self.url_entry.pack(fill="x", padx=16, pady=(8, 0), ipady=6)

        self._section(side, "Settings")
        row = tk.Frame(side, bg=PANEL)
        row.pack(fill="x", padx=16)
        tk.Label(row, text="Confidence", font=(FONT, 10), fg=FG, bg=PANEL).pack(side="left")
        self.conf_label = tk.Label(row, text="0.40", font=(FONT, 10, "bold"), fg=ACCENT, bg=PANEL)
        self.conf_label.pack(side="right")
        self.conf_var = tk.DoubleVar(value=self.conf)
        ttk.Scale(side, from_=0.05, to=0.95, orient="horizontal", variable=self.conf_var,
                  command=self._on_conf).pack(fill="x", padx=16, pady=(2, 8))

        row = tk.Frame(side, bg=PANEL)
        row.pack(fill="x", padx=16, pady=(0, 6))
        tk.Label(row, text="Inference size", font=(FONT, 10), fg=FG, bg=PANEL).pack(side="left")
        self.imgsz_var = tk.StringVar(value=str(self.imgsz))
        box = ttk.Combobox(row, state="readonly", width=7, textvariable=self.imgsz_var,
                           values=IMG_SIZES, font=(FONT, 10))
        box.pack(side="right")
        box.bind("<<ComboboxSelected>>", lambda e: setattr(self, "imgsz", int(self.imgsz_var.get())))
        tk.Label(side, text="Smaller = faster, larger = better for tiny/far objects.",
                 font=(FONT, 8), fg=MUTED, bg=PANEL, anchor="w").pack(fill="x", padx=16)

        self.labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Show labels & scores", variable=self.labels_var,
                        command=lambda: setattr(self, "show_labels", self.labels_var.get())).pack(
            anchor="w", padx=16, pady=(8, 0))

        btns = tk.Frame(side, bg=PANEL)
        btns.pack(fill="x", padx=16, pady=(16, 6))
        self.start_btn = HoverButton(btns, GREEN, "#059669", text="▶  Start Scan", command=self.toggle_scan)
        self.start_btn.pack(fill="x")
        tools = tk.Frame(side, bg=PANEL)
        tools.pack(fill="x", padx=16)
        self.snap_btn = HoverButton(tools, CARD, BORDER, text="📷  Snapshot", command=self.snapshot)
        self.snap_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.rec_btn = HoverButton(tools, CARD, BORDER, text="⏺  Record", command=self.toggle_record)
        self.rec_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self._section(side, "Detections")
        self.det_list = tk.Label(side, text="—", font=("Consolas", 10), fg=FG, bg=CARD, anchor="nw",
                                 justify="left", padx=10, pady=8)
        self.det_list.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        tk.Label(side, text="Space start/stop · S snapshot · R record · F11 fullscreen",
                 font=(FONT, 8), fg=MUTED, bg=PANEL).pack(pady=(0, 10))

    def _build_viewer(self, body):
        view = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        view.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(view, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=1, pady=1)
        self._img_item = self.canvas.create_image(0, 0, anchor="center")
        self._ph_title = self.canvas.create_text(0, 0, text="ROMEO EYE", fill=ACCENT,
                                                 font=(FONT, 34, "bold"))
        self._ph_sub = self.canvas.create_text(0, 0, fill=FG, font=(FONT, 12), justify="center",
                                               text="Choose a model and source, then press Start Scan")
        self._rec_dot = self.canvas.create_text(18, 18, text="", fill=RED, anchor="nw",
                                                font=(FONT, 11, "bold"))
        self.canvas.bind("<Configure>", self._on_resize)

        stats = tk.Frame(view, bg=CARD)
        stats.pack(fill="x")
        self.stat_labels = {}
        for key in ("FPS", "Inference", "Objects", "Resolution"):
            cell = tk.Frame(stats, bg=CARD)
            cell.pack(side="left", padx=18, pady=8)
            tk.Label(cell, text=key.upper(), font=(FONT, 8, "bold"), fg=MUTED, bg=CARD).pack(anchor="w")
            lbl = tk.Label(cell, text="—", font=(FONT, 13, "bold"), fg=FG, bg=CARD)
            lbl.pack(anchor="w")
            self.stat_labels[key] = lbl
        self.status = tk.Label(stats, text="Ready.", font=(FONT, 10), fg=MUTED, bg=CARD, anchor="e")
        self.status.pack(side="right", padx=18)

    def _bind_keys(self):
        self.root.bind("<space>", lambda e: self._key(self.toggle_scan))
        self.root.bind("<KeyPress-s>", lambda e: self._key(self.snapshot))
        self.root.bind("<KeyPress-r>", lambda e: self._key(self.toggle_record))
        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self._fullscreen and self._toggle_fullscreen())

    def _key(self, fn):
        focus = self.root.focus_get()
        if focus is None or focus.winfo_class() not in ("Entry", "TCombobox", "TCheckbutton", "Button"):
            fn()

    # ---------- UI callbacks ----------
    def _on_conf(self, _=None):
        self.conf = round(float(self.conf_var.get()), 2)
        self.conf_label.config(text=f"{self.conf:.2f}")

    def _on_model_change(self, _=None):
        name = self.model_var.get()
        if name == CUSTOM_MODEL:
            path = filedialog.askopenfilename(title="Select YOLO weights",
                                              filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
            if not path:
                self.model_var.set(next(iter(MODEL_CHOICES)))
                return self._on_model_change()
            self.custom_weights = path
            self.model_info.config(text=f"✓ …\\{Path(path).parent.name}\\{Path(path).name}", fg=GREEN)
            return
        path = resolve_weights(name)
        if path:
            self.model_info.config(text=f"✓ …\\{Path(path).parent.name}\\{Path(path).name}", fg=GREEN)
        else:
            self.model_info.config(text="✗ Weights not found. Put the .pt file in the 'weights' folder.", fg=RED)

    def _on_source_change(self, _=None):
        is_ip = self.source_var.get() == "IP Camera (HTTP/RTSP)"
        self.url_entry.config(state="normal" if is_ip else "disabled")

    def _on_resize(self, event):
        self.display_size = (max(event.width, 2), max(event.height, 2))
        cx, cy = event.width // 2, event.height // 2
        self.canvas.coords(self._img_item, cx, cy)
        self.canvas.coords(self._ph_title, cx, cy - 18)
        self.canvas.coords(self._ph_sub, cx, cy + 26)
        if not self.running and self._last_annotated is None:
            self._render_placeholder()
        elif not self.running and self._last_annotated is not None:
            self._show_rgb(self._fit(self._last_annotated))

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _set_status(self, text, color=MUTED):
        self.status.config(text=text, fg=color)

    def _set_controls(self, running):
        state = "disabled" if running else "readonly"
        self.model_box.config(state=state)
        self.source_box.config(state=state)
        if running:
            self.url_entry.config(state="disabled")
            self.start_btn.set_colors(RED, "#dc2626")
            self.start_btn.config(text="■  Stop Scan")
        else:
            self._on_source_change()
            self.start_btn.set_colors(GREEN, "#059669")
            self.start_btn.config(text="▶  Start Scan")

    # ---------- Start / stop ----------
    def toggle_scan(self):
        if self.running:
            self.stop_scan()
        else:
            self.start_scan()

    def _build_source(self):
        s = self.source_var.get()
        if s == "Laptop Camera":
            return 0, "live"
        if s == "Local Video":
            p = filedialog.askopenfilename(title="Select Video", filetypes=[
                ("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.webm"), ("All Files", "*.*")])
            if p and p.lower().endswith(IMAGE_EXTS):
                return p, "image"
            return p, "file"
        if s == "Image File":
            return filedialog.askopenfilename(title="Select Image", filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All Files", "*.*")]), "image"
        url = self.url_entry.get().strip()
        if url.startswith(("http://", "https://", "rtsp://", "rtmp://")) and len(url) > 8:
            return url, "live"
        return "", "live"

    def start_scan(self):
        if self.running:
            return
        name = self.model_var.get()
        weights = self.custom_weights if name == CUSTOM_MODEL else resolve_weights(name)
        if not weights:
            searched = "\n".join(str(p) for p in MODEL_CHOICES.get(name, []))
            messagebox.showerror("Model Error", f"Weights for '{name}' not found. Searched:\n\n{searched}")
            return
        source, kind = self._build_source()
        if source in (None, ""):
            messagebox.showwarning("Source", "No input source selected (or invalid URL).")
            return

        self.running = True
        self._set_controls(True)
        self.canvas.itemconfig(self._ph_title, state="hidden")
        self.canvas.itemconfig(self._ph_sub, state="hidden")
        self._set_status("Loading model…", AMBER)
        self.worker = Thread(target=self._worker, args=(weights, source, kind), daemon=True)
        self.worker.start()

    def stop_scan(self):
        self.running = False
        self._set_status("Stopping…")

    def _finish(self):
        self.running = False
        self.record = False
        self.rec_btn.set_colors(CARD, BORDER)
        self.rec_btn.config(text="⏺  Record")
        self.canvas.itemconfig(self._rec_dot, text="")
        self._set_controls(False)
        self.threat_pill.config(text="● IDLE", fg=MUTED)
        self.stat_labels["FPS"].config(text="—")

    def snapshot(self):
        if self._last_annotated is None:
            return
        SNAP_DIR.mkdir(exist_ok=True)
        path = SNAP_DIR / f"romeo_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        if imwrite_unicode(path, self._last_annotated):
            self._set_status(f"Snapshot saved: {path.name}", GREEN)

    def toggle_record(self):
        if not self.running:
            return
        self.record = not self.record
        if self.record:
            self.rec_btn.set_colors(RED, "#dc2626")
            self.rec_btn.config(text="■  Stop Rec")
            self.canvas.itemconfig(self._rec_dot, text="● REC")
        else:
            self.rec_btn.set_colors(CARD, BORDER)
            self.rec_btn.config(text="⏺  Record")
            self.canvas.itemconfig(self._rec_dot, text="")

    # ---------- Worker thread ----------
    def _post(self, *event):
        self.events.put(event)

    def _worker(self, weights, source, kind):
        cap = reader = writer = None
        try:
            t0 = time.perf_counter()
            model = self.models.get(weights, self.imgsz, lambda msg: self._post("status", msg, AMBER))
            self._post("status", f"Model ready in {time.perf_counter() - t0:.1f}s · {Path(weights).name}", GREEN)

            if kind == "image":
                frame = imread_unicode(source)
                if frame is None:
                    raise RuntimeError(f"Could not read image:\n{source}")
                self._process(model, frame, None)
                return

            if source == 0 and sys.platform == "win32":
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video source:\n{source}")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if not 1 <= src_fps <= 120:
                src_fps = 25.0

            if kind == "live":
                reader = LatestFrameReader(cap)
            seq = 0
            frame_period = 1.0 / src_fps
            next_t = time.perf_counter()
            fps_ema = 0.0
            last = time.perf_counter()

            while self.running:
                if reader:
                    frame, seq = reader.read(seq)
                    if frame is None:
                        if not reader.alive:
                            break
                        continue
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break

                annotated = self._process(model, frame, fps_ema)

                if self.record and writer is None:
                    REC_DIR.mkdir(exist_ok=True)
                    path = REC_DIR / f"romeo_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                    self._post("status", f"Recording → {path.name}", RED)
                elif not self.record and writer is not None:
                    writer.release()
                    writer = None
                    self._post("status", f"Recording saved to {REC_DIR.name}/", GREEN)
                if writer is not None:
                    writer.write(annotated)

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0:
                    fps_ema = 1 / dt if fps_ema == 0 else fps_ema * 0.9 + (1 / dt) * 0.1

                if kind == "file":
                    next_t += frame_period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.perf_counter()
            self._post("status", "Stream ended." if self.running else "Stopped.", MUTED)
        except Exception as e:
            self._post("error", str(e))
        finally:
            if reader:
                reader.stop()
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
                self._post("status", f"Recording saved to {REC_DIR.name}/", GREEN)
            self._post("done")

    def _process(self, model, frame, fps):
        t0 = time.perf_counter()
        res = model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.models.device,
                            verbose=False, **self.models.extra)[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        annotated = res.plot(labels=self.show_labels, conf=self.show_labels)
        names = res.names
        counts = Counter(names[int(c)] for c in res.boxes.cls.tolist()) if res.boxes is not None else Counter()
        self._last_annotated = annotated
        rgb = self._fit(annotated)
        stats = {"fps": fps, "ms": infer_ms, "counts": counts, "res": f"{frame.shape[1]}×{frame.shape[0]}"}
        with self._latest_lock:
            self._latest = (rgb, stats)
        return annotated

    def _fit(self, bgr):
        tw, th = self.display_size
        h, w = bgr.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)

    # ---------- Main-thread rendering ----------
    def _poll(self):
        with self._latest_lock:
            item, self._latest = self._latest, None
        if item is not None:
            rgb, stats = item
            self._show_rgb(rgb)
            now = time.perf_counter()
            if now - self._last_stats_ui > 0.2:
                self._last_stats_ui = now
                self._update_stats(stats)
        while True:
            try:
                ev = self.events.get_nowait()
            except Empty:
                break
            if ev[0] == "status":
                self._set_status(ev[1], ev[2])
            elif ev[0] == "error":
                self._set_status("Error.", RED)
                messagebox.showerror("Romeo Eye", ev[1])
            elif ev[0] == "done":
                self._finish()
        self.root.after(15, self._poll)

    def _show_rgb(self, rgb):
        self._frame_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.itemconfig(self._img_item, image=self._frame_tk)

    def _update_stats(self, s):
        self.stat_labels["FPS"].config(text=f"{s['fps']:.1f}" if s["fps"] else "—")
        self.stat_labels["Inference"].config(text=f"{s['ms']:.0f} ms")
        total = sum(s["counts"].values())
        self.stat_labels["Objects"].config(text=str(total), fg=ACCENT if total else FG)
        self.stat_labels["Resolution"].config(text=s["res"])
        if s["counts"]:
            lines = [f"{name:<16}{n:>4}" for name, n in s["counts"].most_common(12)]
            self.det_list.config(text="\n".join(lines))
        else:
            self.det_list.config(text="No objects above threshold")
        threats = {k: v for k, v in s["counts"].items() if k.lower() in THREAT_CLASSES}
        if threats:
            desc = ", ".join(f"{k} ×{v}" for k, v in threats.items())
            self.threat_pill.config(text=f"▲ THREAT · {desc}", fg=RED)
        else:
            self.threat_pill.config(text="● CLEAR", fg=GREEN)

    def _render_placeholder(self):
        w, h = self.display_size
        if self._bg_src is not None:
            img = self._bg_src.copy()
            scale = max(w / img.width, h / img.height)
            img = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)), Image.BILINEAR)
            left, top = (img.width - w) // 2, (img.height - h) // 2
            img = img.crop((left, top, left + w, top + h))
            img = ImageEnhance.Brightness(img).enhance(0.28)
        else:
            img = Image.new("RGB", (w, h), (10, 10, 10))
        self._placeholder_tk = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self._img_item, image=self._placeholder_tk)
        self.canvas.tag_raise(self._ph_title)
        self.canvas.tag_raise(self._ph_sub)

    def _on_close(self):
        self.running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.root.destroy()


# ---------- Entry ----------
if __name__ == "__main__":
    if sys.stdout is None or sys.stderr is None:  # pythonw has no console
        sys.stdout = sys.stderr = open(os.devnull, "w")
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    app = RomeoEyeApp(root)
    # If URL passed as first argument (e.g. from garrison-sec-dashboard), pre-fill and select IP Camera
    if len(sys.argv) >= 2:
        url = sys.argv[1].strip()
        if url.startswith(("rtsp://", "http://", "https://")):
            app.source_var.set("IP Camera (HTTP/RTSP)")
            app._on_source_change()
            app.url_entry.delete(0, tk.END)
            app.url_entry.insert(0, url)
    # If "visdrone" passed as second argument, pre-select DRONE VIEW (VisDrone) model for drone feed
    if len(sys.argv) >= 3 and sys.argv[2].strip().lower() == "visdrone" and "DRONE VIEW" in MODEL_CHOICES:
        app.model_var.set("DRONE VIEW")
        app._on_model_change()
    root.mainloop()
