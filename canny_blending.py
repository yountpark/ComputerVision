# -*- coding: utf-8 -*-
import cv2
import numpy as np

def nothing():
    pass

def main():
    windowName = 'frame'    
    cv2.namedWindow(windowName)
    cv2.createTrackbar('ALPHA',windowName,0,10,nothing)
    cap = cv2.VideoCapture(0)
    cap.set(3,360)
    cap.set(4,240)
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret= False

    while(ret):
        ret, frame = cap.read()
        frame = cv2.medianBlur(frame,7)
        canny = cv2.Canny(frame,30,70)
        canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
        alpha = cv2.getTrackbarPos('ALPHA',windowName) / 10
        output = cv2.addWeighted(frame,alpha,canny,1-alpha,0)
        cv2.imshow(windowName,np.hstack([frame,canny,output]))
        if cv2.waitKey(1) == 27:
            break;
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
        