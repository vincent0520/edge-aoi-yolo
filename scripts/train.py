from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='configs/neu-det.yaml', epochs=100, imgsz=640, batch=16, device=0, project='runs', name='neu_v8n_baseline', seed=0, patience=50)
