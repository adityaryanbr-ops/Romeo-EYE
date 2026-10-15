import os

def main():
    print("Select Weights:")
    print("1. Original YOLO")
    print("2. Bird vs Drone")
    print("3. SARD Human Detection")
    print("4. VisDrone Pretrained")

    choice = input("Enter choice (1/2/3/4): ")

    if choice == "1":
        weights = "C:/Users/adity/Downloads/yolov8m.pt"
        name = "Original YOLO"
    elif choice == "2":
        weights = "C:/Users/adity/Downloads/drone-detection-starter/runs/detect/birds_vs_drone_a100/weights/best.pt"
        name = "Bird vs Drone"
    elif choice == "3":
        weights = "C:/Users/adity/Downloads/drone-detection-starter/runs/detect/sard_humans_a100/weights/best.pt"
        name = "SARD Human Detection"
    elif choice == "4":
        weights = "C:/Users/adity/Downloads/drone-detection-starter/weights/visdrone_best.pt"
        name = "VisDrone Pretrained"
    else:
        print("Invalid choice. Exiting...")
        return

    print(f"Selected {name} -> {weights}")

    # Input source
    print("\nSelect Input Source:")
    print("1. Laptop Camera")
    print("2. Video (drone_footage.mp4)")
    print("3. Image (car1.jpg)")

    src_choice = input("Enter source (1/2/3): ")

    if src_choice == "1":
        source = "0"
    elif src_choice == "2":
        source = "C:/Users/adity/Downloads/drone_footage.mp4"
    elif src_choice == "3":
        source = "C:/Users/adity/Downloads/car1.jpg"
    else:
        print("Invalid choice. Defaulting to Laptop Camera...")
        source = "0"

    # Run YOLO detection
    command = f"yolo predict model=\"{weights}\" source=\"{source}\" conf=0.4 show=True save=True"
    print(f"Running: {command}")
    os.system(command)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
