"""
Модуль для запису відео з камери та розпізнавання номерних знаків автомобілів.

Цей модуль дозволяє виконувати відеозапис у форматі AVI із збереженням кадрів,
коли режим запису активний (`parameters.GLOBAL_VAR == True`). Під час запису
функція `record_video` також виконує розпізнавання номерних знаків на кадрах
за допомогою бібліотеки `pytesseract`.

Функціонал:
- Запис відео з камери з автоматичною зупинкою після 10 секунд без активності.
- Розпізнавання номерних знаків автомобілів, зокрема, українських номерів
  (валідованих регулярним виразом `UKRAINE_PLATE_PATTERN`).

Основні компоненти:
- Регулярний вираз `UKRAINE_PLATE_PATTERN` для перевірки відповідності
  українським державним номерним знакам.
- Вказівка шляху до виконуваного файлу Tesseract (`pytesseract.pytesseract.tesseract_cmd`).
- Функція `record_video(cap)` для запису відео та розпізнавання номерних знаків.

Залежності:
- `time` — для роботи з часовими мітками.
- `re` — для валідації номерних знаків за допомогою регулярного виразу.
- `numpy` (`np`) — для обробки зображень.
- `cv2` (OpenCV) — для роботи з відео та обробки зображень.
- `pytesseract` — для оптичного розпізнавання тексту.
- `parameters` — модуль із глобальною змінною `parameters.GLOBAL_VAR`,
  яка визначає стан запису.

Примітки:
- Відео зберігається у форматі AVI із часовою міткою в назві, наприклад,
  "car_20241122-1230.avi".
- Якщо номерний знак відповідає регулярному виразу `UKRAINE_PLATE_PATTERN`,
  він виводиться у консоль.
- Ресурси відеозахоплення (`cap.release()`) та запису відео (`out.release()`)
  автоматично звільняються.

Обмеження:
- Підтримуються тільки українські номерні знаки згідно з шаблоном
  `UKRAINE_PLATE_PATTERN`.
- Для коректної роботи Tesseract потрібно налаштувати шлях до виконуваного файлу
  `tesseract.exe`.
"""

import time
import re
import numpy as np
import cv2
import pytesseract
import parameters

pytesseract.pytesseract.tesseract_cmd = r"D:\Python\work\tools\tesseract.exe"

UKRAINE_PLATE_PATTERN = r"^[ABCEHIKMOPTX01]{2}\s?\d{4}\s?[ABCEHIKMOPTX01]{2}$"


def record_video(cap):
    """
       Записує відео з камери, зберігаючи кадри, коли активний режим запису та розпізнає номерні знаки автомобіля.

       Функція обробляє відеопотік, зчитуючи кадри з об'єкта `cap` (cv2.VideoCapture),
       та виконує запис відео у форматі AVI, коли глобальна змінна `parameters.GLOBAL_VAR`
       знаходиться в стані `True`. Під час запису також намагається розпізнати номерний
       знак автомобіля на кадрі.

       Параметри:
       cap (cv2.VideoCapture): Об'єкт відеозахоплення, з якого зчитуються кадри.

       Використання:
       - Передайте відкритий об'єкт `cv2.VideoCapture` для обробки.
       - Запис починається, коли `parameters.GLOBAL_VAR` встановлено в `True`.

       Деталі роботи:
       - Ініціалізує об'єкт `cv2.VideoWriter` при початку запису відео.
       - Використовує алгоритми обробки зображень для визначення номерного знака.
       - Якщо `parameters.GLOBAL_VAR` переходить у стан `False`, запис припиняється через
         10 секунд без активності.

       Збереження:
       - Відео зберігається у форматі AVI із назвою, що включає часову мітку, наприклад,
         "car_20241122-1230.avi".
       - Номерний знак визначається за допомогою бібліотеки `pytesseract` та порівнюється з
         шаблоном `UKRAINE_PLATE_PATTERN`.

       Залежності:
       - `parameters.GLOBAL_VAR` — глобальна змінна для активації режиму запису.
       - `pytesseract` для розпізнавання тексту з номерних знаків.
       - `cv2` (OpenCV) для обробки відео та зображень.
       - Регулярний вираз `UKRAINE_PLATE_PATTERN` для валідації номерних знаків.

       Примітки:
       - Якщо номерний знак розпізнано успішно, він виводиться у консоль.
       - Відеозапис зупиняється автоматично після 10 секунд без активності.
       - Підтримує тільки номерні знаки, що відповідають формату `UKRAINE_PLATE_PATTERN`.

       Завершення:
       - Функція автоматично звільняє ресурси відеозахоплення (`cap.release()`) та запису
         відео (`out.release()`).
       """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None
    plate_number = None
    video_name = None
    stop_time = (
        None
    )

    while True:
        is_recording = parameters.GLOBAL_VAR
        ret, frame = cap.read()
        if not ret:
            break

        if is_recording:
            if out is None:
                video_name = "car_" + time.strftime("%Y%m%d-%H%M") + ".avi"
                out = cv2.VideoWriter(
                    video_name, fourcc, 20.0, (frame.shape[1], frame.shape[0])
                )
            stop_time = None

            out.write(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 200)
            contours, _ = cv2.findContours(
                edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
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
                cropped = masked_image[y : y + h, x : x + w]
                config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                plate_number = pytesseract.image_to_string(
                    cropped, config=config
                ).strip()

                if bool(re.match(UKRAINE_PLATE_PATTERN, plate_number)):
                    print("License plate:", plate_number)

        else:
            if stop_time is None:
                stop_time = time.time() + 10

            if (time.time() > stop_time) and is_recording is False:
                out.release()
                out = None
                stop_time = None

    cap.release()
    if out is not None:
        out.release()
