import cv2

# Відкриваємо відеострім з камери
cap = cv2.VideoCapture(0)  # 0 - перша камера

# Налаштування для запису відео
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для запису
out = None
is_recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Показуємо відео з камери
    cv2.imshow('Camera', frame)

    # Якщо запис увімкнено, зберігаємо поточний кадр
    if is_recording and out is not None:
        out.write(frame)

    # Чекаємо натискання клавіші
    key = cv2.waitKey(1)
    if key == ord('r'):  # Почати/зупинити запис натиснувши 'r'
        if is_recording:
            # Зупиняємо запис
            is_recording = False
            out.release()
            print("Запис завершено.")
        else:
            # Починаємо запис
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            is_recording = True
            print("Запис розпочато.")
    elif key == ord('q'):  # Вихід при натисканні 'q'
        break

# Завершуємо роботу з камерою та закриваємо всі вікна
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
