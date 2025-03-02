import imutils
import cv2

# from the hsv script we get the high and low values
objectLow = (23,10,179)
objectHigh = (55,54,243)

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()

    img = imutils.resize(img, width=600) # reframing
    blurred = cv2.GaussianBlur(img, (11,11), 0) # smoothening
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # converting to hsv
    
    # now we need to mask the color - in our case the s23 phone's white portion
    mask = cv2.inRange(hsv, objectLow, objectHigh)
    
    # we will erode it - it will make it thin and remove holes etc
    mask = cv2.erode(mask, None, iterations=2)
    # then dilating it
    mask = cv2.dilate(mask, None, iterations=2)
    
    # finding contours - to identify boundaries and connect them - making them into a single object
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    center = None # to find the center later
    
    if(len(cnts) > 0): # as we have single colour, so as it will be shown, we will see cnts >0
        c = max(cnts, key=cv2.contourArea)
        
        # drawing minimum enclosure circle around the object
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # we will find the center of the complete object
        # moment is needed
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # setting threshold for the circle
        if radius>10:
            cv2.circle(img, (int(x), int(y)), int(radius), (0,255,255), 2) # drawing circle
            
            # plotting the centroid
            cv2.circle(img, center, 5, (0,0,255), -1) # 5 = fixed radius
            
        # now threshold for maximum limit of circle
        if radius>250:
            print("Stop")
        else:
            if(center[0]<150): # center[0] = x coordinate
                print("Left")
            elif(center[0]>450):
                print("Right")
            elif(radius<250):
                print("Front")
            else:
                print("Stop")
    
    
    # printing the feed
    cv2.imshow("Camera Feed", img)
    
    key = cv2.waitKey(10)
    if (key==27):
        break
    
cam.release()
cv2.destroyAllWindows()