from ultralytics import YOLO
import cv2
import os

TARGET_PATH = "2"
# MODEL_NAME = "weights/yolov8x.pt"
MODEL_NAME = "weights/yolov8-n-best.pt"
# MODEL_NAME = "weights/yolov9-c-best.pt"


def check_name(img_name: str):
    for ext in [".png", ".jpg", "jpeg"]:
        if ext in img_name.lower():
            return True
    return False


def run():
    if not os.path.exists(TARGET_PATH):
        print("Not Exist.")
        return

    model = YOLO(model=MODEL_NAME)
    label_map = model.model.names
    print(len(label_map), label_map)

    img_paths = []
    if not os.path.isdir(TARGET_PATH):
        img_paths = [TARGET_PATH]
    else:
        img_names = os.listdir(TARGET_PATH)
        for img_name in img_names:
            if not check_name(img_name=img_name):
                continue

            img_paths.append(os.path.join(TARGET_PATH, img_name))

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        results = model.predict(img_path)
        print(results)


if __name__ == "__main__":
    print("Detector started")
    run()
