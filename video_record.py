import time
import cv2
import config


def record_video(cap):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    while True:
        is_recording = config.global_var
        ret, frame = cap.read()
        if not ret:
            break

        # Записуємо кадр, якщо запис увімкнено
        if is_recording:
            if out is None:
                video_name = "car_" + time.strftime("%Y%m%d-%H%M") + ".avi"
                # Ініціалізуємо об'єкт запису відео при першому запуску
                out = cv2.VideoWriter(video_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                print("Start recording.")
            out.write(frame)
        else:
            # Зупиняємо запис, якщо він був увімкнений
            if out is not None:
                time.sleep(15)
                out.release()
                out = None
                print("Finish recording.")

        # Завершення роботи при натисканні клавіші 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Завершуємо роботу з камерою та звільняємо ресурси
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


