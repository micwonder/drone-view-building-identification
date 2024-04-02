from ultralytics import YOLO
import cv2
import os
from constants import *
import threading


def check_name(img_name: str):
    for ext in [".png", ".jpg", "jpeg"]:
        if ext in img_name.lower():
            return True
    return False


def box_label(img, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(img.shape) / 2 * 0.0015), 1)  # Line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # Font thickness
        # Calculate text width and height
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        label_p2 = (p1[0] + w, p1[1] - h - 3) if outside else (p1[0] + w, p1[1] + h + 3)
        cv2.rectangle(
            img, p1, label_p2, color, -1, cv2.LINE_AA
        )  # Filled rectangle for label background
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img


def plot_rect(img, boxes, labels=None, colors=None, show_percent=True, confi=0.0):
    if labels is None:
        labels = {0: ""}
    if colors is None:
        colors = COLORS

    labels[0] = "building"  # Default label for index 0, if necessary

    for box in boxes:
        idx = int(box[-1])
        confidence = float(box[-2])
        # if confidence < confi:
        #     continue  # Skip boxes with confidence lower than confi double check after model

        if show_percent:
            label = f"{labels.get(idx, '')} {round(100 * confidence, 1)}%"
        else:
            label = labels.get(idx, "")

        color = colors[
            idx % len(colors)
        ]  # Use modulo for safety in case of index out of range
        img = box_label(img=img, box=box, label=label, color=color)

    return img


# def run_images():
#     if not os.path.exists(TARGET_PATH):
#         print("Not Exist.")
#         return

#     model = YOLO(model=MODEL_NAME)
#     label_map = model.model.names
#     print(len(label_map), label_map)

#     img_paths = []
#     if not os.path.isdir(TARGET_PATH):
#         img_paths = [TARGET_PATH]
#     else:
#         img_names = os.listdir(TARGET_PATH)
#         for img_name in img_names:
#             if not check_name(img_name=img_name):
#                 continue

#             img_paths.append(os.path.join(TARGET_PATH, img_name))

#     window_cnt = 0
#     for img_path in img_paths:
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

#         results = model.predict(img_path)
#         print(results)

#         img = plot_rect(
#             img=img,
#             boxes=results[0].boxes.data,
#             labels=label_map,
#             show_percent=True,
#             # confi=0.5,
#         )
#         height, width, channels = img.shape
#         img = cv2.resize(src=img, dsize=[min(1024, width), min(768, height)])
#         cv2.imshow(f"Drone View{window_cnt % 10}", img)
#         window_cnt += 1

#     cv2.waitKey(0)

#     cv2.destroyAllWindows()


class PredictProcessor:
    def __init__(self, model) -> None:
        self.model = model
        self.frame_to_predict = None
        self.predicted_results = None
        self.lock = threading.Lock()

    def predict(self):
        while True:
            with self.lock:
                if self.frame_to_predict is None:
                    continue
                frame = self.frame_to_predict
                self.frame_to_predict = None

            # Perform prediction
            results = self.model.predict(
                frame, imgsz=IMAGE_SIZE, iou=IOU, conf=CONFIDENCE
            )

            with self.lock:
                self.predicted_results = results


def run_video():
    model = YOLO(model=MODEL_NAME)
    label_map = model.model.names
    # print(len(label_map), label_map)

    processor = PredictProcessor(model=model)
    threading.Thread(target=processor.predict, daemon=True).start()

    # define a video capture object
    vid = cv2.VideoCapture("Train\\thefredric_manual.mp4")
    if not vid.isOpened():
        print("[WARNING] No source found. Looking for source.")
        return
        # lmain.after(10, CheckSource)

    results = None

    while True:

        # Capture the video frame by frame
        ret, frame = vid.read()
        if not ret:
            break

        with processor.lock:
            if (
                processor.frame_to_predict is None
            ):  # Only update if the previous frame has been processed
                processor.frame_to_predict = frame
            if processor.predicted_results is not None:
                results = processor.predicted_results
                processor.predicted_results = None

        if results:
            frame = plot_rect(
                img=frame,
                boxes=results[0].boxes.data,
                labels=label_map,
                show_percent=True,
                # confi=0.3,
            )
        height, width, channels = frame.shape
        frame = cv2.resize(
            src=frame, dsize=[min(WINDOW_WIDTH, width), min(WINDOW_HEIGHT, height)]
        )
        cv2.imshow(f"Drone View", frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Detector started")
    # run_images()
    run_video()
