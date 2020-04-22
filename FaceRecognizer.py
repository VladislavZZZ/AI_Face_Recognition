import os
import glob
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K


def _init_train_model(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))

    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights(weights_path)
    return model


def _init_classifier_model(x_train):
    # Softmax regressor to classify images based on encoding
    classifier_model = Sequential()
    classifier_model.add(Dense(units=100, input_dim=x_train.shape[1], kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.3))
    classifier_model.add(Dense(units=10, kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.2))
    classifier_model.add(Dense(units=6, kernel_initializer='he_uniform'))
    classifier_model.add(Activation('softmax'))
    classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='nadam',
                             metrics=['accuracy'])
    return classifier_model


class FaceRecognizer:

    __WEIGHTS_PATH="/Users/vladislav/PycharmProjects/VGGFace/venv/data/vgg_face_weights.h5"
    __FACE_DETECTION_MODEL_PATH="/Users/vladislav/PycharmProjects/VGGFace/venv/data/mmod_human_face_detector.dat"

    _model = _init_train_model(__WEIGHTS_PATH)
    __vgg_face=Model(inputs=_model.layers[0].input, outputs=_model.layers[-2].output)
    __classifier_model=None

    @staticmethod
    def train_model(train_dataset_path=None,
                    test_dataset_path=None,
                    new_model_directory=not None,
                    epochs=100):
        FaceRecognizer.__train(train_dataset_path, test_dataset_path, new_model_directory, epochs)



    @staticmethod
    def pretrain_model(train_dataset_path=not None,
                       test_dataset_path=None,
                       old_model_path=not None,
                       new_model_directory=not None,
                       epochs=100):
        FaceRecognizer.__classifier_model = tf.keras.models.load_model(old_model_path)
        FaceRecognizer.__train(train_dataset_path,test_dataset_path,new_model_directory,epochs
                               ,FaceRecognizer.__classifier_model)

    @staticmethod
    def recognize(model_path,img):
        FaceRecognizer.__classifier_model = tf.keras.models.load_model(model_path)
        dnn_face_detector = dlib.cnn_face_detection_model_v1(FaceRecognizer.__FACE_DETECTION_MODEL_PATH)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = dnn_face_detector(gray, 1)
        for (i, rect) in enumerate(rects):
            # Extract Each Face
            left = rect.rect.left()  # x1
            top = rect.rect.top()  # y1
            right = rect.rect.right()  # x2
            bottom = rect.rect.bottom()  # y2
            width = right - left
            height = bottom - top
            img_crop = img[top:top + height, left:left + width]
###############################################################################
            cv2.imwrite( 'crop_img.jpg', img_crop)

            # Get Embeddings
            crop_img = load_img('crop_img.jpg', target_size=(224, 224))

  ############################################################################
            crop_img = img_to_array(crop_img)
            crop_img = np.expand_dims(crop_img, axis=0)
            crop_img = preprocess_input(crop_img)
            img_encode = FaceRecognizer.__vgg_face(crop_img)
            # Make Predictions
            embed = K.eval(img_encode)
            person = FaceRecognizer.__classifier_model.predict(embed)
            return np.argmax(person), person[0][np.argmax(person)]

    @staticmethod
    def __train(train_dataset_path=None,
                test_dataset_path=None,
                new_model_directory=not None,
                epochs=100,
                classifier_model=None):
        x_train = []
        y_train = []
        person_folders = os.listdir(train_dataset_path)
        person_rep = dict()
        for i, person in enumerate(person_folders):
            if person == '.DS_Store':
                continue
            person_rep[i] = person
            image_names = os.listdir(train_dataset_path+ person + '/')
            for image_name in image_names:
                if image_name == '.DS_Store':
                    continue
                img = load_img(train_dataset_path + person + '/' + image_name, target_size=(224, 224))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                img_encode = FaceRecognizer.__vgg_face(img)
                x_train.append(np.squeeze(K.eval(img_encode)).tolist())
                y_train.append(i)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

        if classifier_model is None:
            classifier_model = _init_classifier_model(x_train)

        if test_dataset_path is not None:
            x_test = []
            y_test = []
            person_folders = os.listdir(test_dataset_path)
            for i, person in enumerate(person_folders):
                if person == '.DS_Store':
                    continue
                image_names = os.listdir(test_dataset_path + person + '/')
                for image_name in image_names:
                    if image_name == '.DS_Store':
                        continue
                    img = load_img(test_dataset_path + person + '/' + image_name, target_size=(224, 224))
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    img_encode = FaceRecognizer.__vgg_face(img)
                    x_test.append(np.squeeze(K.eval(img_encode)).tolist())
                    y_test.append(i)

            x_test = np.array(x_test)
            y_test = np.array(y_test)
            classifier_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        else:
            classifier_model.fit(x_train, y_train, epochs=epochs)

        tf.keras.models.save_model(classifier_model, new_model_directory)