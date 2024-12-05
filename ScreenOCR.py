import time
import threading
from threading import Thread
import pytesseract
import numpy as np
from PIL import ImageGrab
import cv2
import sys
import keyboard
import argparse


def tesseract_location(root):
    # Sets the Tesseract command root path.
    try:
        pytesseract.pytesseract.tesseract_cmd = root
    except FileNotFoundError:
        print("Please double-check the Tesseract file directory or ensure it's installed.")
        sys.exit(1)


class ScreenCapture:
    # Class to capture frames from the screen in real-time.
    def __init__(self, region=None):
        self.region = region
        self.frame = None
        self.stopped = False

    def start(self):
        # Starts screen capture in a separate thread.
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        # Continuously captures screenshots of the specified region.
        while not self.stopped:
            try:
                screenshot = ImageGrab.grab(bbox=self.region)
                self.frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error during screen capture: {e}")

    def stop_process(self):
        # Stops the screen capture process.
        self.stopped = True


class OCR:
    # Class to perform OCR on captured frames in a separate thread.
    def __init__(self):
        self.boxes = None
        self.stopped = False
        self.capture = None
        self.language = None

    def start(self):
        # Starts OCR processing in a separate thread.
        Thread(target=self.ocr, args=()).start()
        return self

    def set_capture(self, capture):
        # Links the ScreenCapture instance for input frames.
        self.capture = capture

    def set_language(self, language):
        # Sets the language for OCR processing.
        self.language = language

    def ocr(self):
        # Processes frames and extracts text with bounding boxes.
        while not self.stopped:
            if self.capture and self.capture.frame is not None:
                frame = self.capture.frame
                self.boxes = pytesseract.image_to_data(frame, lang=self.language)

    def stop_process(self):
        # Stops the OCR process.
        self.stopped = True


def put_ocr_boxes(boxes, frame):
    # Draws OCR bounding boxes and text on the frame.
    if boxes:
        for i, box in enumerate(boxes.splitlines()):
            if i == 0:
                continue
            box = box.split()
            if len(box) == 12:
                x, y, w, h = map(int, box[6:10])
                try:
                    conf = float(box[10])  # Convert confidence to float
                except ValueError:
                    conf = 0  # Default confidence if conversion fails

                text = box[11]

                color = (0, 255, 0) if conf > 75 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def ocr_screen(region=None, language="eng"):
    # Main function to start OCR on the screen.
    capture = ScreenCapture(region=region).start()
    ocr = OCR().start()

    ocr.set_capture(capture)
    ocr.set_language(language)

    # Initialize the scanning region
    x1, y1, x2, y2 = region if region else (0, 0, 800, 600)
    edit_mode = False  # Toggle for resizing mode

    try:
        while True:
            # Toggle edit mode with the "e" key
            if keyboard.is_pressed("e"):
                edit_mode = not edit_mode
                time.sleep(0.2)  # Prevent rapid toggling

            if edit_mode:
                # Adjust the region dynamically using hotkeys
                if keyboard.is_pressed("up"):
                    y1 = max(0, y1 - 10)
                if keyboard.is_pressed("down"):
                    y1 += 10
                if keyboard.is_pressed("left"):
                    x1 = max(0, x1 - 10)
                if keyboard.is_pressed("right"):
                    x1 += 10
                if keyboard.is_pressed("w"):
                    y2 = max(y1 + 10, y2 - 10)
                if keyboard.is_pressed("s"):
                    y2 += 10
                if keyboard.is_pressed("a"):
                    x2 = max(x1 + 10, x2 - 10)
                if keyboard.is_pressed("d"):
                    x2 += 10

                # Update the capture region
                capture.region = (x1, y1, x2, y2)

            if capture.frame is not None:
                # Only capture the region specified in ScreenCapture
                frame = capture.frame.copy()

                if ocr.boxes:
                    frame = put_ocr_boxes(ocr.boxes, frame)

                # Draw the blue or yellow rectangle to indicate the scanning area
                if edit_mode:
                    cv2.rectangle(frame, (0, 0), (x2 - x1, y2 - y1), (0, 255, 255), 2)  # Yellow border in edit mode
                    cv2.putText(frame, "EDIT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (0, 0), (x2 - x1, y2 - y1), (255, 0, 0), 2)  # Blue border in normal mode

                cv2.imshow("Screen OCR", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key press.
                break
    finally:
        capture.stop_process()
        ocr.stop_process()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse arguments for language and Tesseract path
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tess_path", help="Path to the Tesseract executable",
                        default="C:/Program Files/Tesseract-OCR/tesseract.exe")
    parser.add_argument("-l", "--language", help="Language code for Tesseract (default: eng)", default="eng")
    args = parser.parse_args()

    # Set Tesseract location
    tesseract_location(args.tess_path)

    # Start OCR with specified language
    ocr_screen(region=(0, 0, 800, 600), language=args.language)
