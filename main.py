import cv2
import threading
from video_record import record_video
from car_recognition import detect_cars_from_video
from car_license_plates_recognition import recognize_license_plate_from_video


def main():
    cap = cv2.VideoCapture(0)

    record_thread = threading.Thread(target=record_video, args=(cap,))
    detect_thread = threading.Thread(target=detect_cars_from_video, args=(cap,))
    recognize_thread = threading.Thread(target=recognize_license_plate_from_video, args=(cap,))

    record_thread.start()
    detect_thread.start()
    recognize_thread.start()

    record_thread.join()
    detect_thread.join()
    recognize_thread.join()


if __name__ == "__main__":
    main()
