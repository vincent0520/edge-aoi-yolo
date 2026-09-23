import fiftyone as fo
import fiftyone.zoo as foz

def main():
    classes = ["Bottle","Cup","Book","Laptop","Chair","Car","Cat","Dog",
               "Person","Bicycle","Bird","Clock","Flower","Table","Backpack"]
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=50000,
        seed=51,
    )
    dataset.export(
        export_dir="datasets/openimages_yolo",
        dataset_type=fo.types.YOLOv5Dataset,
        classes=classes,
    )
    print("Done. Exported to datasets/openimages_yolo")

if __name__ == "__main__":
    main()