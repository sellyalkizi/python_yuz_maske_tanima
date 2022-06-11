# -*- coding: utf-8 -*-
# gerekli paketleri içe aktar
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


#yüz dedektör modelini diskten yükle
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# yüz maskesi dedektör modelini diskten yükle
print("[INFO] loading face mask detector model...")
model = load_model("modelim.model")

# girdi görüntüsünü diskten yükle, kopyala ve görüntünün uzamsal boyutlarını al
image = cv2.imread("images/20.jpg")
orig = image.copy()
(h, w) = image.shape[:2]

# görüntüden bir blob oluştur
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# blobu ağ üzerinden geçir ve yüz algılamalarını al
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# algılamalar üzerinde döngü
for i in range(0, detections.shape[2]):
# algılamayla ilişkili güveni (yani olasılığı) çıkar
	confidence = detections[0, 0, i, 2]
# güvenin minimum güvenden daha büyük olmasını sağlayarak zayıf algılamaları filtrele
	if confidence > 0.5:
# nesne için sınırlayıcı kutunun (x, y)-koordinatlarını hesapla
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
        # sınırlayıcı kutuların çerçevenin boyutları dahilinde olduğundan emin ol
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # yüz ROI'sini çıkarın, BGR'den RGB kanal sıralamasına dönüştür, 
        # 224x224'e yeniden boyutlandırın ve önişle
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

# yüzün maskeli olup olmadığını belirlemek için yüzü modelden geçirin
		(undermask,mask, withoutMask) = model.predict(face)[0]
		print(undermask,mask,withoutMask)

 # sınırlayıcı kutuyu ve metni çizmek için kullanacağımız sınıf etiketini ve rengini belirleyin

		if withoutMask>0.8:
			label = "Maske Yok"
		elif mask>0.8:
			label = "Maske Var"
		else:
			label = "Maske Altta"


		if (label=="Maske Var"):
 			color = (0, 255, 0) 
		elif (label=="Maske Yok"): 
 			color = (0, 0, 255)
		elif(label=="Maske Altta"): 
 			color = (255, 0, 0)


# etikete olasılığı dahil et
		label = "{}: {:.2f}%".format(label, max(undermask, mask, withoutMask) * 100)

# etiket ve sınırlayıcı kutu dikdörtgenini çıktı çerçevesinde görüntüle
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# çıktı görüntüsünü göster
cv2.imshow("Output", image)
cv2.waitKey(0)