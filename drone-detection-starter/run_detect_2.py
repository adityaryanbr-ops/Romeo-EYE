import os

def main():
    print("Select Model Weights:")
    print("1. Original YOLOv8m")
    print("2. Bird vs Drone")
    print("3. SARD Human Detection")

    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        weights = r"C:\Users\adity\Downloads\yolov8m.pt"
        name = "Original YOLOv8m"
    elif choice == "2":
        weights = r"C:\Users\adity\Downloads\drone-detection-starter\runs\detect\birds_vs_drone_a100\weights\best.pt"
        name = "Bird vs Drone"
    elif choice == "3":
        weights = r"C:\Users\adity\Downloads\drone-detection-starter\runs\detect\sard_humans_a100\weights\best.pt"
        name = "SARD Human Detection"
    else:
        print("Invalid choice. Exiting...")
        return

    print(f"Selected {name} -> {weights}")

    # Input Source Selection
    print("\nSelect Input Source:")
    print("1. Laptop Camera")
    print("2. Video (drone_footage.mp4)")
    print("3. Image (test.jpg)")

    src_choice = input("Enter source (1/2/3): ")

    if src_choice == "1":
        source = "0"
    elif src_choice == "2":
        source = "drone_footage.mp4"
    elif src_choice == "3":
        source = "test.jpg"
    else:
        print("Invalid choice. Defaulting to Laptop Camera...")
        source = "0"

    # Build the YOLO detect command
    command = f"python src\\inference\\detect.py --weights \"{weights}\" --source {source} --conf 0.4 --show --save"
    print(f"\nRunning command: {command}")

    # Run the detection
    os.system(command)


if __name__ == "__main__":
    main()
