from ultralytics import YOLO

def main():
    for name in ["yolov8m", "yolov8l", "yolov8x"]:
        print(f"===== Training {name} =====")
        model = YOLO(f"{name}.pt")
        model.train(data="configs/mvtec.yaml", epochs=100, imgsz=640,
                    batch=16, device=0, project="runs",
                    name=f"mvtec_{name}_baseline", seed=0, patience=50, verbose=False)

if __name__ == "__main__":
    main()