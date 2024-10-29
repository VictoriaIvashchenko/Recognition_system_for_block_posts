import cv2
import torch
from ultralytics import YOLO
import config
import time


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
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                # Малюємо рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Оновлюємо глобальну змінну
        config.global_var = car_detected

        # Відображаємо кадр з детекцією
        cv2.imshow("Car Detection", frame)

        # Зупиняємо, якщо натиснута клавіша 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

