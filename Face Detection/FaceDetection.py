import cv2

#alg = "haarcascade_frontalface_default.xml"

# Loading library using cv2
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
while True:
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    # face will give values like x,y,w,h
    
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if(key==27): # 21 is for escape
        break
    
cam.release()
cv2.destroyAllWindows()