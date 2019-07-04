# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path
#from operator import itemgetter
import operator
pathList = "C:\\Users\\gowns\\.spyder-py3\\carNum"
#for item in os.listdir(pathList):
#    print(item)
#print(os.listdir(pathList)[0])
#kernel = np.ones((5,5),np.uint8)
global original_image
def gaussianF(image,kernel):
    gaussianB_image = cv2.GaussianBlur(image,(kernel,kernel),0)
    return gaussianB_image
def histogram(image):
    equal_histogram = cv2.equalizeHist(image)
    return equal_histogram
def substraction(histogram_image,morph_image):
    sub_morp_image = cv2.subtract(histogram_image,morph_image)
    return sub_morp_image
def dilated(image):
    kernel = np.ones((3,3),np.uint8)
    dilated_image = cv2.dilate(image,kernel,iterations=1)
    return dilated_image
def perspective(pos, img):
    height, width = original_image.shape[:2]
    height = int(height/4)
    width = int(width/3)
    
    
    set_pos = np.float32([pos[0][0],pos[1][0],pos[2][0],pos[3][0]])
    pts = np.float32([[0,0],[0,height],[width,height],[width,0]])

    perspective_img = cv2.getPerspectiveTransform(set_pos,pts)
    img_result = cv2.warpPerspective(img,perspective_img,(width,height))
    cv2.namedWindow("window 2",cv2.WINDOW_NORMAL)
    cv2.imshow("window 2",img_result)
def contour(dilated_image,original_copy):
    contours,hierarchy = cv2.findContours(dilated_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]

    original_copy1 = np.copy(gray_image)
    original_copy2 = np.copy(original_image)
    
    #비어있는 배열 전처리
    original_copy1 = cv2.Canny(original_copy1,255,255)
    ret,original_copy1 = cv2.threshold(original_copy1,255,255,cv2.THRESH_BINARY)
        
    # 비어있는 배열 만들기
    (cnts,_) = cv2.findContours(original_copy1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    screenCnt=None
    
    for c in contours:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02*peri,True)
        
        if len(approx)==4:
            cnts.append(approx)
        
        
    return cnts
    
def option1(gray_image):
    #S02
    gaussian_image = gaussianF(gray_image,5)
    histogram_image = histogram(gaussian_image)
    ret,thresh_image = cv2.threshold(histogram_image,110,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    morph_image = cv2.morphologyEx(thresh_image,cv2.MORPH_OPEN,kernel,iterations=15)
    sub_image = substraction(thresh_image,morph_image)
    screenCnt = contour(sub_image,original_image)
    #print("---------------------screenCnt2------------------")
    #print(screenCnt)
    return screenCnt
def option2(gray_image):
    #S05
    gaussian_image = gaussianF(gray_image,5)
    ret,thresh_image = cv2.threshold(gaussian_image,80,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph_image = cv2.morphologyEx(thresh_image,cv2.MORPH_OPEN,kernel,iterations=15)
    sub_image = substraction(thresh_image,morph_image)
    screenCnt = contour(sub_image, original_image)
    #print("---------------------screenCnt3------------------")
    #print(screenCnt)
    return screenCnt
    
def option3(gray_image):
    #S06
    gaussian_image = gaussianF(gray_image,5)
    histogram_image = histogram(gaussian_image)
    ret,thresh_image = cv2.threshold(histogram_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    screenCnt = contour(thresh_image,original_image)
    #print("---------------------screenCnt4------------------")
    #print(screenCnt)
    return screenCnt
    
def option4(gray_image):
    #S08, S01, S03, S04
    gaussian_image = gaussianF(gray_image,5)
    ret,thresh_image = cv2.threshold(gaussian_image,120,255,cv2.THRESH_BINARY)
    #histogram_image = histogram(gaussian_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph_image = cv2.morphologyEx(gaussian_image,cv2.MORPH_OPEN,kernel,iterations=15)
    sub_image = substraction(thresh_image,morph_image)
    screenCnt = contour(sub_image,original_image)
    #print("---------------------screenCnt5------------------")
    #print(screenCnt)
    return screenCnt
def option5(gray_image):
    #S07
    gaussian_image = gaussianF(gray_image,5)
    ret,thresh_image = cv2.threshold(gaussian_image,175,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    morph_image = cv2.morphologyEx(gaussian_image,cv2.MORPH_OPEN,kernel,iterations=25)
    sub_image = substraction(thresh_image,morph_image)
    screenCnt = contour(sub_image,original_image)
    return screenCnt
def checkRectangle(cnts,original_copy):
    flag = False
    screenCnt = None
    result_image = original_copy
    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02*peri,True)
        convecx_check = cv2.isContourConvex(approx)
        #print(" ----------------------check ----------------------")
        if (len(approx)==4 and convecx_check!=False):
            #print(approx)
            x,y,w,h = cv2.boundingRect(c)
            ratio = (float)(h)/w
            print("ratio:%f" %ratio)
            screenCnt = approx
            #print(type(screenCnt))
            if(ratio>=0.13 and ratio <=0.45 and h<w):
                flag = True
                cv2.rectangle(original_copy,(x,y),(x+w,y+h),(0,0,255),3)
                rect = cv2.minAreaRect(c)
                box=cv2.boxPoints(rect)
                box = np.int0(box)
                result_image=cv2.drawContours(original_copy,[box],0,(0,255,0),1)
                print("다시한번 더 들어와썽")
                break
    
    if(flag==False):
        #번호판을 찾지 못했다는 것을 알려줌
        no= np.zeros((130*3, 130*4), np.uint8)
        cv2.putText(no,"NOT FOUND",(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.namedWindow("window2")
        cv2.imshow('window2',no)
        
    return screenCnt
    
while(True):
    print('시작')
    start = input('엔터키 입력')
    try:
        if start == '':
            for i in range(0,5):
                print(os.listdir(pathList)[i])
                original_image = cv2.imread(pathList+"/"+os.listdir(pathList)[i])
                if original_image is None:
                    error = np.zeros((1000,1000),np.uint8)
                    cv2.putText(error,"No file", (50,300), cv.FONT_HERSHEY_SIMPLEX,4,(255,255,255),2)
                    cv2.namedWindow("window 2")
                    cv2.show('window 2',error)
                    key = cv2.waitKey(0)
                    if key==27:
                        break;
                    elif key==13:
                        continue
                original_image = cv2.resize(original_image, dsize=(int(original_image.shape[1]/6),
                                                                   int(original_image.shape[0]/8)),
                                            interpolation=cv2.INTER_AREA)
                height,width = original_image.shape[:2]
                
                #print("height: %d width: %d" %(height,width))
                cv2.namedWindow("window 1",cv2.WINDOW_NORMAL)
                cv2.imshow("window 1",original_image)
                gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
                gray_copy1 = np.copy(gray_image)
                original_copy1 = np.copy(gray_image)
                original_copy2 = np.copy(original_image)
                original_copy_result = np.copy(original_image)
                
                #비어있는 배열 전처리
                original_copy1 = cv2.Canny(original_copy1,255,255)
                ret,original_copy1 = cv2.threshold(original_copy1,255,255,cv2.THRESH_BINARY)
                
                
                # 비어있는 배열 만들기
                (cnts,_) = cv2.findContours(original_copy1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

                #print(option1(gray_image))
                cnts.extend(option1(gray_image))
                cnts.extend(option2(gray_image))
                cnts.extend(option3(gray_image))
                cnts.extend(option4(gray_image))
                cnts.extend(option5(gray_image))
                cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]
                #print("------------------------cnts---------------------")
                #print(cnts)
                pos = checkRectangle(cnts,original_copy2)
                #print(pos)
               
                for i in range(4):
                    for j in range(3):
                        if pos[j][0][0] > pos[j+1][0][0]:
                                tmp1 = pos[j+1][0][0]
                                tmp2 = pos[j+1][0][1]
                                pos[j+1][0][0] = pos[j][0][0]
                                pos[j+1][0][1] = pos[j][0][1]
                                pos[j][0][0] = tmp1
                                pos[j][0][1] = tmp2
                                

                if pos[0][0][1] > pos[1][0][1]:
                    tmp1 = pos[1][0][0]
                    tmp2 = pos[1][0][1]
                    pos[1][0][0] = pos[0][0][0]
                    pos[1][0][1] = pos[0][0][1]
                    pos[0][0][0] = tmp1
                    pos[0][0][1] = tmp2

                if pos[3][0][1] > pos[2][0][1]:
                    tmp1 = pos[3][0][0]
                    tmp2 = pos[3][0][1]
                    pos[3][0][0] = pos[2][0][0]
                    pos[3][0][1] = pos[2][0][1]
                    pos[2][0][0] = tmp1
                    pos[2][0][1] = tmp2
                
                

                #print("--------------")
                #print(pos)
                perspective(pos, original_copy_result)
                while True:
                    key = cv2.waitKey(0)
                    if key == 13:
                        continue
                    elif key == 27:
                        exit()
            
                        
        break
    except:
        error = np.zeros((130*3,130*4),np.uint8)
        cv2.putText(error,"ERROR", (50,300), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
        cv2.imshow('window 2',error)
cv2.destroyAllWindows()
        
    
