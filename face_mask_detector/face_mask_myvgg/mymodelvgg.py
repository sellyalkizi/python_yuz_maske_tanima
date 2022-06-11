# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from modelvgg.mymodelvggnet import MyModelVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import scikitplot.metrics as splt
import numpy as np
import random
import cv2
import os
import pickle
# verileri ve etiketleri başlat
print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images('C:/Users/selca/face-mask-detector/dataset3')))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
 	image = cv2.imread(imagePath)
 	image = cv2.resize(image, (64,64))
 	data.append(image)
 	label = imagePath.split(os.path.sep)[-2]
 	labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
#Verilerin %75'ini eğitim için ve kalan %25'ini test için kullanarak
#verileri eğitim ve test bölümlerine ayırın
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# veri büyütme için görüntü oluşturucuyu oluştur
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True,vertical_flip=True, fill_mode="nearest")


# VGG benzeri Evrişimsel Sinir Ağımızı başlat
model = MyModelVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))


# ilk öğrenme oranını ve eğitilecek dönemlerin sayısını başlat
INIT_LR = 0.01
EPOCHS = 75
BS=64

print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model.summary()

# Ağı Eğit
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# Ağı Değerlendir
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)

splt.plot_confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
plt.savefig('C:/Users/selca/face-mask-detector/output/confusplt.png')

print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

#eğitim kaybını ve doğruluğunu çiz
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('C:/Users/selca/face-mask-detector/output/plot.png')

# modeli kaydet
print("[INFO] serializing network and label binarizer...")
model.save('C:/Users/selca/face-mask-detector/output/mymodelvgg.model',save_format="h5")
f = open('C:/Users/selca/face-mask-detector/output/simple_nn_lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()
