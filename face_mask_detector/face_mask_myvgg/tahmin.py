# Gerekli Kütüphaneleri Ekle
from keras.models import load_model
import pickle
import cv2
# Giriş görüntüsünü yükle ve yeniden boyutandır
image = cv2.imread("images/20.jpg")
output = image.copy()
image = cv2.resize(image, (64, 64))

# piksel değerlerini [0, 1] olarak ölçeklendir
image = image.astype("float") / 255.0

if -1 > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

print("[INFO] loading network and label binarizer...")
model = load_model("output/mymodelvgg.model")
lb = pickle.loads(open("output/simple_nn_lb.pickle", "rb").read())

preds = model.predict(image)
print(preds)
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

if label=="with_mask": color = (0, 255, 0) 
elif label=="without_mask": color = (0, 0, 255)
else: color = (255, 0, 0)

text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	color, 2)

cv2.imshow("Image", output)
cv2.waitKey(0)