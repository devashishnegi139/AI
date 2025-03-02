import cv2
import os # for checking and using directory

dataset = "dataset" # our database folder
name = "Devashish" # Particular person folder name

path = os.path.join(dataset, name) # dataset/Devashish - like this dir will be formed
# checking if path is available or not
if not os.path.isdir(path):
    # os.mkdirs(path)  # will create the folder, if not available
    os.makedirs(path) # will create the entire tree dir

(width, height) = (130,100) # w and h of image to be saved

#--------------------------------------------------------------

#alg = "haarcascade_frontalface_default.xml"
# Loading library using cv2
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

count = 1

while count<31: # we want just 30 images
    print(count)
    
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    # face will give values like x,y,w,h
    
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        faceOnly = grayImg[y:y+h, x:x+w] # will give us face only image
        resizeImg = cv2.resize(faceOnly, (width, height))
        cv2.imwrite("%s/%s.jpg"%(path,count), resizeImg)
        
        count+=1
        
        
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if(key==27): # 21 is for escape
        break
print("Image Captured Successfully!")    
cam.release()
cv2.destroyAllWindows()