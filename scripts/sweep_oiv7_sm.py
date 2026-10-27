from ultralytics import YOLO

def main():
    for name in ["yolov8x"]:
        print(f"===== Training {name} on Open Images =====")
        model = YOLO(f"{name}.pt")
        model.train(
            data="configs/openimages.yaml",
            epochs=100, imgsz=640, batch=16, device=0,
            project="runs", name=f"oiv7_50k_{name}",
            seed=0, patience=15, verbose=False,
        )
    print("Open Images s/m done.")

if __name__ == "__main__":
    main()