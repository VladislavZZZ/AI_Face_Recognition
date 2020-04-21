from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import cv2

class ImageDetector:

    def __init__(self):
        self.__Detector=MTCNN()

    def detect_faces(self,img_path):
        img = pyplot.imread(img_path)
        results =self.__Detector.detect_faces(img)
        return results


    def show_detected_faces(self,results,img_path):
        img = pyplot.imread(img_path)
        for res in results:
            x, y, w, h = res['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
