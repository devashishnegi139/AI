# Step 1 - capture feed
# Step 2 - capture 1st image to fix background image
# Step 3 - compare b/w background image and current image

import cv2
import time
import imutils
import numpy as np

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 5000

while True:
    _, img = cam.read()

    # Preprocessing Image
    text = "Normal"
    img = imutils.resize(img, width=300)  # Resize to smaller width to fit screen
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  # Apply Gaussian blur
    
    # Step 2: Capture first frame for background
    if firstFrame is None: 
        firstFrame = gaussianImg
        continue
    
    # Step 3: Compute difference between first frame and current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    thresImg = cv2.threshold(imgDiff, 50, 255, cv2.THRESH_BINARY)[1]
    thresImg = cv2.dilate(thresImg, None, iterations=2)
    
    # Find contours
    cnts = cv2.findContours(thresImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected"
        
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Convert grayscale images to BGR for concatenation
    grayImg_bgr = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
    gaussianImg_bgr = cv2.cvtColor(gaussianImg, cv2.COLOR_GRAY2BGR)
    imgDiff_bgr = cv2.cvtColor(imgDiff, cv2.COLOR_GRAY2BGR)
    thresImg_bgr = cv2.cvtColor(thresImg, cv2.COLOR_GRAY2BGR)
    
    # Resize all images to the same size for proper display
    img_resized = cv2.resize(img, (400, 300))
    gaussian_resized = cv2.resize(gaussianImg_bgr, (400, 300))
    imgDiff_resized = cv2.resize(imgDiff_bgr, (400, 300))
    thres_resized = cv2.resize(thresImg_bgr, (400, 300))
    
    # Stack images in 2x2 grid (mosaic)
    top_row = np.hstack((img_resized, gaussian_resized))   # stack original and Gaussian images
    bottom_row = np.hstack((imgDiff_resized, thres_resized)) # stack difference and threshold images
    mosaic = np.vstack((top_row, bottom_row))     # stack rows vertically

    # Display the mosaic
    cv2.imshow("All Feed", mosaic)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): # exit if 'q' is pressed
        break

cam.release()
cv2.destroyAllWindows()
