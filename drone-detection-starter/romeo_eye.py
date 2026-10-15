import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2, os
from ultralytics import YOLO

# ---------- Background image ----------
BG_MERGED = r"C:\Users\adity\Downloads\drone-detection-starter\assets\romeo_bg.jpg"

# ---------- Model paths (adjust if needed) ----------
MODEL_CHOICES = {
    "General": r"C:\Users\adity\Downloads\yolov8m.pt",
    "Drone Detection": r"C:\Users\adity\Downloads\drone-detection-starter\runs\detect\birds_vs_drone_a100\weights\best.pt",
    "OPERATIONAL DPLY": r"C:\Users\adity\Downloads\drone-detection-starter\runs\detect\sard_humans_a100\weights\best.pt",
    "DRONE VIEW": r"C:\Users\adity\Downloads\yolov8m-visdrone\best.pt",
    "WEAPON DETECTION": r"C:\Users\adity\Downloads\drone-detection-starter\assets\gunmodel\best3.pt",
}

SOURCE_CHOICES = ["Laptop Camera", "Local Video", "Image File", "IP Camera (HTTP/RTSP)"]

# ---------- Theme ----------
BG_COLOR  = "#0b0f16"
PANEL_BG  = "#101724"
TITLE_FONT = ("Segoe UI", 32, "bold")
SUB_FONT   = ("Segoe UI", 12)
LABEL_FONT = ("Segoe UI", 12, "bold")
TEXT_FONT  = ("Segoe UI", 11)
BTN_FONT   = ("Segoe UI", 13, "bold")

def multiline_size(draw: ImageDraw.ImageDraw, text: str, font):
    """Pillow-compatible multiline text measure (works on older Pillow)."""
    lines = text.splitlines() or [text]
    max_w, total_h = 0, 0
    line_spacing = 6
    for i, line in enumerate(lines):
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(line, font=font)
        max_w = max(max_w, w)
        total_h += h + (line_spacing if i < len(lines) - 1 else 0)
    return max_w, total_h

class RomeoEyeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROMEO EYE")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1140x760")

        self.model = None
        self.running = False
        self.cap = None
        self.display_size = (1040, 540)  # (w, h)
        self._frame_tk = None
        self._bg_tk = None

        self._ensure_merged_background()
        self._place_background()

        self.container = tk.Frame(self.root, bg=BG_COLOR)
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)

        tk.Label(self.container, text="ROMEO EYE", font=TITLE_FONT,
                 fg="#22d3ee", bg=BG_COLOR).pack(pady=10)
        tk.Label(self.container, text="Real-time Aerial Detection Control Panel",
                 font=SUB_FONT, fg="white", bg=BG_COLOR).pack(pady=(0, 10))

        # Controls row 1
        panel = tk.Frame(self.container, bg=PANEL_BG); panel.pack(pady=6, padx=20)

        left = tk.Frame(panel, bg=PANEL_BG); left.grid(row=0, column=0, padx=12, pady=8, sticky="w")
        tk.Label(left, text="Type of detection model:", font=LABEL_FONT, fg="white", bg=PANEL_BG).pack(anchor="w")
        self.model_var = tk.StringVar(value=list(MODEL_CHOICES.keys())[0])
        ttk.Combobox(left, state="readonly", textvariable=self.model_var, width=40,
                     values=list(MODEL_CHOICES.keys()), font=TEXT_FONT).pack(pady=4)

        mid = tk.Frame(panel, bg=PANEL_BG); mid.grid(row=0, column=1, padx=12, pady=8, sticky="w")
        tk.Label(mid, text="Source:", font=LABEL_FONT, fg="white", bg=PANEL_BG).pack(anchor="w")
        self.source_var = tk.StringVar(value=SOURCE_CHOICES[0])
        ttk.Combobox(mid, state="readonly", textvariable=self.source_var, width=28,
                     values=SOURCE_CHOICES, font=TEXT_FONT).pack(pady=4)

        right = tk.Frame(panel, bg=PANEL_BG); right.grid(row=0, column=2, padx=12, pady=8, sticky="w")
        tk.Label(right, text="IP/HTTP/RTSP URL (if selected):", font=LABEL_FONT, fg="white", bg=PANEL_BG).pack(anchor="w")
        self.url_entry = tk.Entry(right, width=42, font=TEXT_FONT)
        self.url_entry.insert(0, "http://")
        self.url_entry.pack(pady=4)

        # Controls row 2: confidence + buttons (same row so you always see them)
        strip = tk.Frame(self.container, bg=BG_COLOR); strip.pack(pady=6)
        tk.Label(strip, text="Confidence:", font=LABEL_FONT, fg="white", bg=BG_COLOR).pack(side="left", padx=(0,10))
        self.conf_var = tk.DoubleVar(value=0.40)
        ttk.Scale(strip, from_=0.10, to=0.80, orient="horizontal", length=320,
                  variable=self.conf_var).pack(side="left", padx=(0,16))

        self.start_btn = tk.Button(strip, text="Start Scan", font=BTN_FONT, bg="#10b981", fg="white",
                                   activebackground="#059669", activeforeground="white",
                                   padx=16, pady=6, command=self.start_scan)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = tk.Button(strip, text="Stop Scan", font=BTN_FONT, bg="#ef4444", fg="white",
                                  activebackground="#dc2626", activeforeground="white",
                                  padx=16, pady=6, state="disabled", command=self.stop_scan)
        self.stop_btn.pack(side="left", padx=8)

        # Video area
        canvas_wrap = tk.Frame(self.container, bg=PANEL_BG); canvas_wrap.pack(pady=12)
        self.canvas = tk.Label(canvas_wrap, bg="black", width=self.display_size[0], height=self.display_size[1])
        self.canvas.pack()
        self._render_placeholder()

        self.status = tk.Label(self.container, text="Ready.", anchor="w", fg="white", bg=BG_COLOR, font=TEXT_FONT)
        self.status.pack(fill="x", padx=24, pady=(6, 12))

    # ----- background helpers -----
    def _ensure_merged_background(self):
        try:
            if not (os.path.exists(BG_LEFT) and os.path.exists(BG_RIGHT)): return
            left = Image.open(BG_LEFT).convert("RGB")
            right = Image.open(BG_RIGHT).convert("RGB")
            h = max(left.height, right.height)
            left = left.resize((int(left.width*h/left.height), h), Image.LANCZOS)
            right = right.resize((int(right.width*h/right.height), h), Image.LANCZOS)
            merged = Image.new("RGB", (left.width+right.width, h))
            merged.paste(left, (0,0)); merged.paste(right, (left.width,0))
            merged = merged.resize((1140,760), Image.LANCZOS)
            merged.save(BG_MERGED, quality=92)
        except Exception:
            pass

    def _place_background(self):
        try:
            if os.path.exists(BG_MERGED):
                img = Image.open(BG_MERGED).convert("RGB")
                self._bg_tk = ImageTk.PhotoImage(img)
                tk.Label(self.root, image=self._bg_tk, bd=0).place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            pass

    # ----- source picking -----
    def _pick_video(self):
        return filedialog.askopenfilename(title="Select Video",
            filetypes=[("Video Files","*.mp4;*.avi;*.mov;*.mkv"), ("All Files","*.*")])

    def _pick_image(self):
        return filedialog.askopenfilename(title="Select Image",
            filetypes=[("Image Files","*.jpg;*.jpeg;*.png;*.bmp"), ("All Files","*.*")])

    def _build_source(self):
        s = self.source_var.get()
        if s == "Laptop Camera": return 0
        if s == "Local Video":   return self._pick_video()
        if s == "Image File":    return self._pick_image()
        if s == "IP Camera (HTTP/RTSP)":
            url = self.url_entry.get().strip()
            return url if url.startswith(("http://","https://","rtsp://")) else ""
        return None

    # ----- start/stop -----
    def start_scan(self):
        if self.running: return
        weights = MODEL_CHOICES.get(self.model_var.get())
        if not weights or not os.path.exists(weights):
            messagebox.showerror("Model Error", f"Weight file not found:\n{weights}")
            return
        source = self._build_source()
        if source in (None, ""):
            messagebox.showwarning("Source", "No input source selected.")
            return

        try:
            self.model = YOLO(weights)
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model:\n{e}")
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text=f"Running: {self.model_var.get()} | Source: {self.source_var.get()}")

        if isinstance(source, str) and source.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            Thread(target=self._run_image, args=(source,), daemon=True).start()
        else:
            Thread(target=self._run_stream, args=(source,), daemon=True).start()

    def stop_scan(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text="Ready.")

    # ----- inference loops -----
    def _run_image(self, path):
        try:
            res = self.model.predict(source=path, conf=float(self.conf_var.get()))
            frame = res[0].plot()
            self._render_frame(frame)
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
        finally:
            self.stop_scan()

    def _run_stream(self, source):
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.status.config(text="Unable to open video source.")
                self.stop_scan(); return
            while self.running:
                ok, frame = self.cap.read()
                if not ok: break
                res = self.model(frame, conf=float(self.conf_var.get()))
                annotated = res[0].plot()
                self._render_frame(annotated)
        except Exception as e:
            messagebox.showerror("Stream Error", str(e))
        finally:
            if self.cap: self.cap.release(); self.cap = None
            self.stop_scan()

    # ----- rendering -----
    def _render_placeholder(self):
        w, h = self.display_size
        img = Image.new("RGB", (w, h), (10, 10, 10))
        draw = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            f = ImageFont.load_default()
        msg = "Video will appear here.\nChoose model & source then click Start Scan."
        tw, th = multiline_size(draw, msg, f)
        draw.multiline_text(((w - tw)//2, (h - th)//2), msg, fill=(200, 200, 200), font=f, align="center")
        self._frame_tk = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._frame_tk)

    def _render_frame(self, bgr):
        tw, th = self.display_size
        h, w = bgr.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (tw - nw)//2, (th - nh)//2
        framed = cv2.copyMakeBorder(resized, pad_y, th - nh - pad_y, pad_x, tw - nw - pad_x,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        rgb = cv2.cvtColor(framed, cv2.COLOR_BGR2RGB)
        self._frame_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=self._frame_tk)

# ---------- Entry ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = RomeoEyeApp(root)
    root.mainloop()
