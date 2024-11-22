"""
Модуль для паралельного запуску запису відео та детекції автомобілів у реальному часі.

Цей модуль координує виконання двох основних завдань:
1. Запис відео з камери із розпізнаванням номерних знаків автомобілів
   (функція `record_video` з модуля `video_record`).
2. Детекція автомобілів у кадрі за допомогою моделі YOLO
   (функція `detect_cars_from_video` з модуля `car_recognition`).

Реалізація:
- Для паралельного виконання використовується модуль `threading`, який забезпечує
  багатопотоковість.
- Потоки створюються для кожної задачі (`record_video` і `detect_cars_from_video`)
  та синхронізуються через `join()`.

Залежності:
- `threading` — для створення та управління потоками.
- `cv2` — для роботи з відеозахопленням.
- `video_record.record_video` — модуль і функція для запису відео.
- `car_recognition.detect_cars_from_video` — модуль і функція для детекції автомобілів.

Примітки:
- Потоки працюють одночасно, забезпечуючи обробку відео в реальному часі.
- Об'єкт відеозахоплення (`cv2.VideoCapture`) передається як аргумент в обидві функції.

"""
import threading
import cv2
from video_record import record_video
from car_recognition import detect_cars_from_video


def main():
    """
        Основна функція для запуску потоків запису відео та детекції автомобілів.

        Функція виконує наступне:
        1. Ініціалізує об'єкт відеозахоплення `cv2.VideoCapture`.
        2. Створює два потоки:
           - Один для запису відео з розпізнаванням номерних знаків (`record_video`).
           - Інший для детекції автомобілів у реальному часі (`detect_cars_from_video`).
        3. Запускає обидва потоки.
        4. Очікує завершення роботи потоків через `join()`.

        Використання:
        - Функція призначена для запуску через `if __name__ == "__main__":`.

        Примітки:
        - Потоки працюють одночасно, обробляючи один і той же відеопотік.
        - Об'єкт `cv2.VideoCapture` звільняється автоматично після завершення обробки.

        """
    cap = cv2.VideoCapture(0)

    record_thread = threading.Thread(target=record_video, args=(cap,))
    detect_thread = threading.Thread(target=detect_cars_from_video, args=(cap,))

    record_thread.start()
    detect_thread.start()

    record_thread.join()
    detect_thread.join()


if __name__ == "__main__":
    """
        Точка входу в програму.

        Викликає основну функцію `main()` для запуску потоків запису відео 
        та детекції автомобілів у реальному часі.
        """
    main()
