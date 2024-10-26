import cv2
import pytesseract
import re


# Вказуємо шлях до виконуваного файлу Tesseract (для Windows)
pytesseract.pytesseract.tesseract_cmd = r'D:\Python\work\tools\tesseract.exe'

# Регулярний вираз для українських номерів
ukraine_plate_pattern = re.compile(r'[А-Я]{2}\s?\d{4}\s?[А-ЯA-Z]{2}')


def recognize_ukrainian_license_plate(image_path, output_file='license_plates.txt'):
    # Завантажуємо зображення
    image = cv2.imread(image_path)

    # Попередня обробка зображення
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Перетворення в чорно-біле
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Розмиття для зменшення шумів
    edged = cv2.Canny(blur, 30, 200)  # Пошук контурів

    # Знаходимо контури
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plates = []

    for contour in contours:
        # Перевіряємо контури на можливі номерні знаки
        approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:  # Імовірно прямокутник
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y + h, x:x + w]  # Вирізаємо регіон із номерним знаком
            # Розпізнавання тексту
            plate_text = pytesseract.image_to_string(roi, config='--psm 8', lang='ukr')
            plate_text = plate_text.strip().replace(" ", "")
            print(plate_text)
            # Перевірка на відповідність українському шаблону
            match = ukraine_plate_pattern.search(plate_text)
            if match:
                plates.append(match.group())

    # Записуємо номери у файл
    with open(output_file, 'w') as file:
        for plate in plates:
            if plate:
                file.write(plate + '\n')

    print(f"Номери збережено у файл {output_file}")


# Виклик функції
recognize_ukrainian_license_plate('plate4.jpg')