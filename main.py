import cv2
import numpy as np 
from tensorflow.keras.models import load_model

model = load_model('Digit Recognition Model.h5')
np.set_printoptions(suppress=True)
img = cv2.imread('8.jpg')

img = cv2.resize(img, (640, 480))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_array = cv2.bitwise_not(img_gray)
img_array = cv2.resize(img_array, (28, 28))
img_array = img_array.reshape((-1, 28, 28, 1))

prediction = model.predict(img_array)[0]
number = np.argmax(prediction)
confidence = int(prediction[number] * 100)

cv2.putText(img, f"{str(number)}, {str(confidence)}%", (25, 120), cv2.FONT_HERSHEY_PLAIN, 7, (0, 255, 0), 4)

cv2.imshow("Image", img)
cv2.waitKey(0)
