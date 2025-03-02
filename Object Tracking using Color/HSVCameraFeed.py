import cv2

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    cv2.imshow("Camera Feed", hsv)
    
    key = cv2.waitKey(10)
    if (key==27):
        break

cam.release()
cv2.destroyAllWindows()