import cv2 
import numpy as np
import sys

#사용할 3개 파일(1. 빨간버스, 2. 학관 위에 초록색 지붕? 3. 본관위에 동그란 초록색
#C:\\Users\\gowns\\.spyder-py3\\FILE0002_van_2017_04_07.avi
#C:/Users/gowns/.spyder-py3/1st_school_tour_studentcenter.avi
#C:/Users/gowns/.spyder-py3/1st_school_tour_headquarter.avi
    
def main():
    path='C:/Users/gowns/.spyder-py3/1st_school_tour_headquarter.avi'
    windowName='frame'
    dis=40
    res=None
    MIN_MATCH_COUNT = 5
    FLANN_INDEX_KDTREE = 0
    #동영상 로드 및 해상도 조절
    cap = cv2.VideoCapture(path) 
    cap.set(3,360)
    cap.set(4,240)
    
    #tracker = cv2.MultiTracker_create()

    #동영상 파일이 열렸는지 확인
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
        print("open 실패")
    
    frame = cv2.resize(frame, dsize=(480,360), interpolation=cv2.INTER_AREA)
    frame_copy = np.copy(frame)
    cv2.namedWindow(windowName)
    cv2.imshow(windowName,frame)
    # 사용자가 원하는 부분을 지정할 수 있게 해줌
    # 처음 x,y,width,height 나옴
    rect = cv2.selectROI(windowName,frame,fromCenter=False,showCrosshair=True)
    x,y,width,height = rect
    newX = np.int32((x + x+width)/2)
    newY = np.int32((y + y+height)/2)
    print(rect)
    cv2.destroyWindow(windowName)

    # 사각형으로 객체를 추출하기 위한 top, bottom, left, right의 좌표
    result_top = y
    result_bottom = (int)(abs(y+height))
    result_left = x
    result_right = (int)(abs(x+width))
    print("top: %d bottom: %d left: %d right: %d" %(result_top,result_bottom,result_left,result_right))
    result_img = frame_copy[result_top:result_bottom, result_left:result_right,:]
    #tracker 세팅(사용자가 지정한 부분(rect)이 frame 변화마다 따라갈 수 있도록 만듬)
    sift = cv2.xfeatures2d.SIFT_create()
    
    while(ret):
        # 동영상 읽어오기
        ret, frame = cap.read()
        if(ret==False):
            break
        #동영상 frame 사이즈 조절
        frame = cv2.resize(frame, dsize=(480,360), interpolation=cv2.INTER_AREA)
    
        cv2.circle(frame,(newX,newY),dis,(0,255,255),2)
        #cv2.rectangle(frame,pt1=(x,y), pt2=(x+width,y+height),color=(255,255,255),thickness=3)
        kp1, des1 = sift.detectAndCompute(result_img,None)
        kp2, des2 = sift.detectAndCompute(frame,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        
        good=[]
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)            
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            
            matchesMask = mask.ravel().tolist()

            h,w = result_img.shape[:2]
            #pts = np.float32([[x,y],[x+width,y],[x+width,y+height],[x,y+height]]).reshape(-1,1,2)
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            print("*****dst******")
         
            newX = np.int32((dst[0][0][0]+dst[2][0][0])/2) #center
            newY = np.int32((dst[0][0][1]+dst[2][0][1])/2) #center
            
            #cv2.rectangle(frame,pt1=(dst[0][0][0],dst[0][0][1]), pt2=(dst[2][0][0],dst[2][0][1]),
            #              color=(255,255,255),thickness=3)
            
            result_img = cv2.polylines(result_img,[np.int32(dst)],True, 255,3,cv2.LINE_AA)
            cv2.imshow('dst',result_img)
        else:
            print("not enough matches")
            matchesMask = None

        res = cv2.drawMatches(result_img,kp1,frame,kp2,good,res,
                                       singlePointColor = (255,0,0),
                              matchColor=(0,0,255),flags=0)
        #cv2.imshow('test',result_img)
        #cv2.imshow(windowName,frame)
        cv2.imshow(windowName,res)

        dst = dst.astype(int)
        result_top = dst[0][0][1]
        result_bottom = (int)(abs(dst[0][0][1]+height))
        result_left = dst[0][0][0]
        result_right = (int)(abs(dst[0][0][0]+width))
        
        result_img = frame_copy[result_top:result_bottom, result_left:result_right,:]

        print("top: %d bottom: %d left: %d right: %d" %(result_top,result_bottom,result_left,result_right))

        if cv2.waitKey(400) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()
        
    
