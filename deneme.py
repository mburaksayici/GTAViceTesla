import cv2
import numpy as np


img = cv2.imread('C:/Users/hp/Desktop/gtavicecityaraba/gtares.jpg')

print("a")

img = cv2.convertScaleAbs(img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )


lower_yellow = np.array([11, 105, 95])
upper_yellow = np.array([142, 183, 168])

lower_white = np.array([20, 75, 100])
upper_white = np.array([30, 100, 120])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask2 = cv2.inRange(hsv, lower_white, upper_white)

combinedmask = cv2.bitwise_or(mask, mask2)



res = cv2.bitwise_and(img, img, mask=combinedmask)


while True:
    cv2.imshow('e',res)

    cv2.waitKey()



 #   processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
  #  vertices = np.array([[0, 300], [0, 640], [30, 640], [30, 300]], np.int32)

