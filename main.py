# example of face detection with mtcnn
# from matplotlib import pyplot
# from PIL import Image
# from numpy import asarray
# from mtcnn.mtcnn import MTCNN
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #filename = "venv/1.jpeg"
# # extract a single face from a given photograph
# def extract_face(filename, required_size=(416, 416)):
# 	# load image from file
# 	pixels = pyplot.imread(filename)
# 	# create the detector, using default weights
# 	detector = MTCNN()
# 	# detect faces in the image
# 	results = detector.detect_faces(pixels)
# 	# extract the bounding box from the first face
# 	x1, y1, width, height = results[0]['box']
# 	x2, y2 = x1 + width, y1 + height
# 	# extract the face
# 	face = pixels[y1:y2, x1:x2]
# 	# resize pixels to the model size
# 	image = Image.fromarray(face)
# 	image = image.resize(required_size)
# 	image.save("ex.jpg")
# 	face_array = asarray(image)
# 	return face_array
#
# # load the photo and extract the face
# pixels = extract_face('/Users/vladislav/PycharmProjects/VGGFace/myplot.jpg')
#
# # plot the extracted face
# pyplot.imshow(pixels)
# # show the plot
# pyplot.show()

from matplotlib import pyplot
from ImageDetector import ImageDetector
from FaceRecognizer import FaceRecognizer

# img=pyplot.imread('/Users/vladislav/Desktop/TrainSandBox/forTrain/111.jpg')
# res = ImageDetector.detect_faces(img)
# ImageDetector.show_detected_faces(res,img)

person_rep = {0: "Vlad", 2: "Other"}

img = pyplot.imread("/Users/vladislav/Desktop/TrainSandBox/forTrain/images/photo_2020-04-07 19.59.57.jpeg")

ans = FaceRecognizer.recognize("/Users/vladislav/PycharmProjects/VGGFace/venv/model/face_classifier_model.h5", img)

print(person_rep[ans[0]], ans[1] * 100)
