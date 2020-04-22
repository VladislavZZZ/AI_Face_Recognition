from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import cv2

class ImageDetector:
    __Detector=MTCNN()

    @staticmethod
    def detect_faces(img):
        results =ImageDetector.__Detector.detect_faces(img)
        return results

    @staticmethod
    def show_detected_faces(results,img):
        for res in results:
            x, y, w, h = res['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
