import cv2
import time

cam = cv2.VideoCapture(0)
time.sleep(1)
while True:
    _, img = cam.read()
    cv2.imshow("Camera Feed", img)
    key = cv2.waitKey(1) & 0XFF # 0XFF will give us 8bit value
    if key==ord("q"): # when person press q, it will break
        break

cam.release()
cv2.destroyAllWindows()