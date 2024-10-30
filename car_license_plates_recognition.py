import cv2
import pytesseract
import re
import config

# Вказуємо шлях до виконуваного файлу Tesseract (для Windows)
pytesseract.pytesseract.tesseract_cmd = r'D:\Python\work\tools\tesseract.exe'

# Регулярний вираз для українських номерів
ukraine_plate_pattern = re.compile(r'[АВСЕНІКМОРТХ]{2}\s?\d{4}\s?[АВСЕНІКМОРТХ]{2}')


def recognize_license_plate_from_video(cap):
    if(config.global_var):
        plate_number = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("The video is over or the frame could not be read.")
                break

            # Попередня обробка кадру
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Перетворення в чорно-біле
            blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Розмиття для зменшення шумів
            edged = cv2.Canny(blur, 30, 200)  # Пошук контурів

            # Знаходимо контури
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Перевіряємо контури на можливі номерні знаки
                approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)

                if len(approx) == 4:  # Імовірно прямокутник
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = frame[y:y + h, x:x + w]  # Вирізаємо регіон із номерним знаком

                    # Розпізнавання тексту
                    plate_text = pytesseract.image_to_string(roi, config='--psm 8')
                    plate_text = plate_text.strip().replace(" ", "")

                    # Перевірка на відповідність українському шаблону
                    match = ukraine_plate_pattern.search(plate_text)
                    if match:
                        plate_number = match.group()
                        print(f"Recognized plate number: {plate_number}")
                        cap.release()
                        cv2.destroyAllWindows()
                        return plate_number

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if plate_number is None:
            print("License plate not found.")
        return plate_number

