import time
import cv2
import pytesseract
import numpy as np
import re
import parameters

# Вказуємо шлях до виконуваного файлу Tesseract (для Windows)
pytesseract.pytesseract.tesseract_cmd = r'D:\Python\work\tools\tesseract.exe'

# Регулярний вираз для українських номерів
ukraine_plate_pattern = r"^[ABCEHIKMOPTX01]{2}\s?\d{4}\s?[ABCEHIKMOPTX01]{2}$"


def record_video(cap):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    plate_number = None
    video_name = None
    stop_time = None  # Час, коли потрібно завершити запис після зміни is_recording на False

    while True:
        is_recording = parameters.global_var
        ret, frame = cap.read()
        if not ret:
            break

        # Якщо почався запис, ініціалізуємо об'єкт запису
        if is_recording:
            if out is None:
                video_name = "car_" + time.strftime("%Y%m%d-%H%M") + ".avi"
                out = cv2.VideoWriter(video_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                # print("Start recording.")
            stop_time = None  # Скидаємо таймер завершення запису, якщо recording знову увімкнений

            out.write(frame)

            # Визначаємо номерний знак, якщо його ще не розпізнано
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 200)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            license_plate = None

            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
                if len(approx) == 4:
                    license_plate = approx
                    break

            if license_plate is not None:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [license_plate], -1, 255, -1)
                masked_image = cv2.bitwise_and(gray, gray, mask=mask)
                (x, y, w, h) = cv2.boundingRect(license_plate)
                cropped = masked_image[y:y + h, x:x + w]
                config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                plate_number = pytesseract.image_to_string(cropped, config=config).strip()

                if bool(re.match(ukraine_plate_pattern, plate_number)):
                    print("License plate:", plate_number)

        else:
            # Запускаємо таймер на 10 секунд при відключенні запису
            if stop_time is None:
                stop_time = time.time() + 10

            # Якщо 10 секунд пройшло після вимкнення запису, зупиняємо запис
            if (time.time() > stop_time) and is_recording is False:
                out.release()
                out = None
                # print("Finish recording.")
                stop_time = None  # Скидаємо таймер завершення запису

    # Завершуємо роботу з камерою та звільняємо ресурси
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()



