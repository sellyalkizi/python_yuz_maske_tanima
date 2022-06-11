# -*- coding: utf-8 -*-
# gerekli paketleri içe aktar
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pickle

def detect_and_predict_mask(frame, faceNet, maskNet):

    # çerçevenin boyutlarını alın ve ondan bir damla oluştur
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
    
    # blobu ağ üzerinden geçirin ve yüz algılamalarını al
	faceNet.setInput(blob)
	detections = faceNet.forward()

# yüzler listemizi, bunlara karşılık gelen konumları ve
# yüz maskesi ağımızdan tahminlerin listesini başlat
	faces = []
	locs = []
	preds = []


# algılamalar üzerinde döngü
	for i in range(0, detections.shape[2]):

        # algılamayla ilişkili güveni (yani olasılığı) çıkar

		confidence = detections[0, 0, i, 2]

        # güvenin minimum güvenden daha büyük olmasını sağlayarak zayıf algılamaları filtrele
		if confidence > args["confidence"]:

            # nesne için sınırlayıcı kutunun (x, y)-koordinatlarını hesapla
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

            # sınırlayıcı kutuların çerçevenin boyutları dahilinde olduğundan emin ol
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            # yüz ROI'sini çıkarın, BGR'den RGB kanal sıralamasına dönüştür,
            # 224x224'e yeniden boyutlandır ve önişle
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (64,64))
			face = img_to_array(face)
			face = preprocess_input(face)

            # yüz ve sınırlayıcı kutuları ilgili listelerine ekle
            
			faces.append(face)
			locs.append((startX, startY, endX, endY))

    # yalnızca en az bir yüz algılandığında tahminde bulun
	if len(faces) > 0:

        # daha hızlı çıkarım için yukarıdaki "for" döngüsündeki tek tek tahminler yerine
        #tüm yüzler üzerinde aynı anda toplu tahminler yap
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

    # yüz konumlarının 2 demetini ve bunlara karşılık gelen konumları döndür
	return (locs, preds)

# argüman ayrıştırıcısını oluştur ve argümanları ayrıştır
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="output/sonn7.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
# serileştirilmiş yüz dedektör modelimizi diskten yükleyin
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
# yüz maskesi dedektör modelini diskten yükleyin
print("[INFO] loading face mask detector model...")
maskNet = load_model("output/mymodelvgg.model")

lb = pickle.loads(open("output/simple_nn_lb.pickle", "rb").read())
# video akışını başlatın ve kamera sensörünün ısınmasına izin verin
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# video akışındaki kareler üzerinde döngü
while True:
# akıtılan video akışından çerçeveyi alın ve maksimum 400 piksel
# genişliğe sahip olacak şekilde yeniden boyutlandır
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

    # çerçevedeki yüzleri algıla ve yüz maskesi takıp takmadıklarını belirle
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

# algılanan yüz konumları ve bunlara karşılık gelen konumlar üzerinde döngü
	for (box, pred) in zip(locs, preds):

        	# sınırlayıcı kutuyu ve tahminleri aç
		(startX, startY, endX, endY) = box
		(undermask, mask, withoutMask) = pred
		print(undermask,mask,withoutMask)

		if undermask>mask and undermask>withoutMask:
 			label = "Maske Altta"
		elif withoutMask>mask and withoutMask>undermask:
 			label = "Maske Yok"
		elif mask>withoutMask and mask>undermask:
 			label = "Maske Var"
		else:
 			label = "Yüz Algılanmadı"

		if (label=="Maske Var"):
 			color = (0, 255, 0) 
		elif (label=="Maske Yok"): 
 			color = (0, 0, 255)
		elif(label=="Maske Altta"): 
 			color = (255, 0, 0)
             
		label = "{}: {:.2f}%".format(label, max( undermask, mask, withoutMask) * 100)

# etiket ve sınırlayıcı kutu dikdörtgenini çıktı çerçevesinde görüntüle
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

# çıktı çerçevesini göster
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

    
# `q` tuşuna basılmışsa, döngüden çık
	if key == ord("q"):
		break

#temizlik yap
vs.release()
cv2.destroyAllWindows()
vs.stop()