import cv2
import imutils

img = cv2.imread("1.png")
resizeImg = imutils.resize(img, width=200)
cv2.imwrite("ResizedImage.png", resizeImg)