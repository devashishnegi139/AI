import cv2
import time

cam = cv2.VideoCapture(0)
time.sleep(1)

_, img = cam.read() # .read() returns two values - flag and image
cv2.imwrite("ImageFromWebcam.png", img)

# we need to release the camera too
cam.release()