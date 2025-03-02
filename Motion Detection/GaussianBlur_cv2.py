import cv2

img = cv2.imread("1.png")
gaussianBlurImg = cv2.GaussianBlur(img, (21,21), 0)
cv2.imwrite("Gaussian Blur Image.png", gaussianBlurImg)