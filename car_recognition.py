import cv2
from ultralytics import YOLO
import parameters


def detect_cars_from_video(cap):
    # Завантажуємо модель YOLO
    model = YOLO("yolov8n.pt")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Виконуємо детекцію об'єктів на кадрі
        results = model(frame)

        # Перевіряємо результати детекції
        car_detected = False
        for result in results[0].boxes:
            class_id = int(result.cls)
            if class_id == 2:  # Індекс класу для автомобілів у COCO — 2
                car_detected = True

        # Оновлюємо глобальну змінну
        parameters.global_var = car_detected

    cap.release()

