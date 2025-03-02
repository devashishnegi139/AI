# for thresholding we need GrayScale image
import cv2
img = cv2.imread("1.png")
grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("GrayScale Image.png", grayScaleImg)

thresImg = cv2.threshold(grayScaleImg, 150, 255, cv2.THRESH_BINARY)[1]
# returns multiple values, therefore we used [1]

cv2.imwrite("Threshold Image.png", thresImg)