import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2

# Predefined weights
WEIGHTS = {
    "Original YOLO": "C:/Users/adity/Downloads/yolov8m.pt",
    "Bird vs Drone": "C:/Users/adity/Downloads/drone-detection-starter/runs/detect/birds_vs_drone_a100/weights/best.pt",
    "SARD Human Detection": "C:/Users/adity/Downloads/drone-detection-starter/runs/detect/sard_humans_a100/weights/best.pt",
    "VisDrone Pretrained": "C:/Users/adity/Downloads/yolov8m-visdrone/best.pt"
}

def run_detection():
    try:
        # Get selected weight
        weight_name = weight_var.get()
        weight_path = WEIGHTS[weight_name]

        # Get source
        source_type = source_var.get()
        if source_type == "Laptop Camera":
            source = 0
        elif source_type == "Video File":
            source = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        elif source_type == "Image File":
            source = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        elif source_type == "IP/HTTP Stream":
            source = ip_entry.get()
        else:
            messagebox.showerror("Error", "Invalid source selected")
            return

        if not source and source != 0:
            messagebox.showerror("Error", "No source selected!")
            return

        # Load YOLO model
        model = YOLO(weight_path)

        # Run detection
        results = model.predict(source=source, conf=0.4, show=True, save=True)

        messagebox.showinfo("Done", f"Detection complete! Results saved in: {results[0].save_dir}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("YOLOv8 Detection GUI")

tk.Label(root, text="Select Weights:").pack(pady=5)
weight_var = tk.StringVar(value="Original YOLO")
tk.OptionMenu(root, weight_var, *WEIGHTS.keys()).pack(pady=5)

tk.Label(root, text="Select Source:").pack(pady=5)
source_var = tk.StringVar(value="Laptop Camera")
tk.OptionMenu(root, source_var, "Laptop Camera", "Video File", "Image File", "IP/HTTP Stream").pack(pady=5)

tk.Label(root, text="(Optional) Enter IP/HTTP Stream URL:").pack(pady=5)
ip_entry = tk.Entry(root, width=40)
ip_entry.pack(pady=5)

tk.Button(root, text="Run Detection", command=run_detection, bg="green", fg="white").pack(pady=10)

root.mainloop()
