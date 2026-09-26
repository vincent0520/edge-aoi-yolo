from ultralytics import YOLO

def main():
    for name in ["yolov8m", "yolov8l", "yolov8x"]:
        print(f"===== Training {name} on NEU-DET =====")
        model = YOLO(f"{name}.pt")
        model.train(
            data="configs/neu-det.yaml",
            epochs=100, imgsz=640, batch=16, device=0,
            project="runs", name=f"neu_{name}_baseline",
            seed=0, patience=50, verbose=False,
        )
    print("NEU-DET m/l/x done.")

if __name__ == "__main__":
    main()