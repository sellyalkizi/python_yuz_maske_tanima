# -*- coding: utf-8 -*-
# gerekli paketleri içe aktarma
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import scikitplot.metrics as splt
import random
import numpy as np
import os
#başlangıç öğrenme oranını, eğitilecek dönem sayısını ve parti boyutunu başlat
INIT_LR = 0.01
EPOCHS = 5
BS = 10
# veri seti dizinimizdeki görüntülerin listesini al, ardından
#veri listesini (yani, görüntüler) ve sınıf görüntülerini başlat
print("[INFO] loading images...")
imagePaths = list(paths.list_images('C:/Users/selca/face-mask-detector/dataset2'))
random.seed(42)
random.shuffle(imagePaths)
data = []
labels = []
# görüntü yolları üzerinde döngü
for imagePath in imagePaths:
# giriş görüntüsünü (224x224) yükle ve önişle
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
# dosya adından sınıf etiketini çıkar
	label = imagePath.split(os.path.sep)[-2]
# sırasıyla veri ve etiket listelerini güncelle
	data.append(image)
	labels.append(label)
# verileri ve etiketleri NumPy dizilerine dönüştürün
data = np.array(data, dtype="float32")
labels = np.array(labels)
# eğitim için verilerin %75'ini ve test için kalan %25'i kullanarak
# verileri eğitim ve test bölümlerine ayır
(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.25, stratify=labels, random_state=42)
# etiketler üzerinde one-hot kodlama gerçekleştir
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# veri büyütme için eğitim görüntüsü oluşturucuyu oluştur
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")
# MobileNetV2 ağını yükle
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# temel modelin üstüne yerleştirilecek modelin başını oluştur
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)
# kafamodelini temel modelin üstüne yerleştir (bu, eğiteceğimiz asıl model olacak)
model = Model(inputs=baseModel.input, outputs=headModel)
# temel modeldeki tüm katmanlar üzerinde döngü yap ve onları dondur,
#böylece ilk eğitim sürecinde güncellenmezler
for layer in baseModel.layers:
	layer.trainable = False
# modelimizi derle
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# ağın başını eğit
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
# test setinde tahminlerde bulun
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# test setindeki her bir görüntü için, karşılık gelen en büyük tahmin edilen
# olasılığa sahip etiketin indeksini bulmamız gerekiyor
predIdxs = np.argmax(predIdxs, axis=1)
# güzel biçimlendirilmiş bir sınıflandırma raporu göster
#Sınıf İsimleri = ['under_mask', 'with_mask', 'without_mask']
splt.plot_confusion_matrix(testY.argmax(axis=1), predIdxs)
plt.show()
plt.savefig('C:/Users/selca/face-mask-detector/confusionplt.png')
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# modeli diske serileştir kaydet
print("[INFO] saving mask detector model...")
model.save('C:/Users/selca/face-mask-detector/modelim.model', save_format="h5")
# eğitim kaybını ve doğruluğunu çiz
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('C:/Users/selca/face-mask-detector/cizimim.png')
plt.show()