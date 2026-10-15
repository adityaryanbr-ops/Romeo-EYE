import tkinter as tk
from tkinter import filedialog, messagebox
import os

def run_detection():
    weight_choice = weight_var.get()
    source_choice = source_var.get()

    # Map weights
    if weight_choice == "Original YOLO":
        weights = "C:/Users/adity/Downloads/yolov8m.pt"
    elif weight_choice == "Bird vs Drone":
        weights = "C:/Users/adity/Downloads/drone-detection-starter/runs/detect/birds_vs_drone_a100/weights/best.pt"
    elif weight_choice == "SARD Human Detection":
        weights = "C:/Users/adity/Downloads/drone-detection-starter/runs/detect/sard_humans_a100/weights/best.pt"
    elif weight_choice == "VisDrone Pretrained":
        weights = "C:/Users/adity/Downloads/yolov8m-visdrone/best.pt"
    else:
        messagebox.showerror("Error", "Invalid weight selection")
        return

    # Map sources
    if source_choice == "Laptop Camera":
        source = "0"
    elif source_choice == "Local Video":
        source = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4")])
    elif source_choice == "Image File":
        source = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.png")])
    elif source_choice == "IP Camera (HTTP/RTSP)":
        source = url_entry.get().strip()
        if not source:
            messagebox.showerror("Error", "Enter an IP camera URL")
            return
    else:
        messagebox.showerror("Error", "Invalid source selection")
        return

    if not source:
        messagebox.showwarning("Warning", "No source selected!")
        return

    # Build YOLO command
    command = f"yolo predict model={weights} source={source} conf=0.4 show=True save=True"
    print("Running:", command)
    os.system(command)


# ------------------ GUI ------------------
root = tk.Tk()
root.title("YOLO Detection GUI")
root.geometry("600x400")
root.configure(bg="#1e1e2e")

# Title Label
title = tk.Label(root, text="YOLOv8 Detection System", font=("Arial", 16, "bold"), bg="#1e1e2e", fg="white")
title.pack(pady=10)

# Frame for Weights
frame1 = tk.LabelFrame(root, text=" Select Model Weights ", padx=10, pady=10, bg="#2e2e3e", fg="white")
frame1.pack(pady=10, fill="x")

weight_var = tk.StringVar(value="Original YOLO")
weights_menu = tk.OptionMenu(frame1, weight_var, "Original YOLO", "Bird vs Drone", "SARD Human Detection", "VisDrone Pretrained")
weights_menu.config(width=25)
weights_menu.pack()

# Frame for Sources
frame2 = tk.LabelFrame(root, text=" Select Input Source ", padx=10, pady=10, bg="#2e2e3e", fg="white")
frame2.pack(pady=10, fill="x")

source_var = tk.StringVar(value="Laptop Camera")
sources_menu = tk.OptionMenu(frame2, source_var, "Laptop Camera", "Local Video", "Image File", "IP Camera (HTTP/RTSP)")
sources_menu.config(width=25)
sources_menu.pack()

# Entry for IP Camera URL
url_label = tk.Label(frame2, text="IP Camera URL:", bg="#2e2e3e", fg="white")
url_label.pack()
url_entry = tk.Entry(frame2, width=40)
url_entry.pack()

# Run Button
run_btn = tk.Button(root, text="Run Detection", font=("Arial", 12, "bold"), bg="#4caf50", fg="white", command=run_detection)
run_btn.pack(pady=20)

root.mainloop()
