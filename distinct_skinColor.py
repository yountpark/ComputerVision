import cv2
import numpy as np



img = cv2.imread("/Users/yount/Desktop/0.png",1)

img = cv2.resize(img, dsize=(int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA) 
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_w = np.array([0, 2, 110], dtype = "uint8")
upper_w = np.array([20, 122, 230], dtype = "uint8")
lower_b = np.array([2,90,25],dtype="uint8")
upper_b = np.array([15,250,130],dtype="uint8") 
skinMask_w = cv2.inRange(hsv,lower_w,upper_w)
skinMask_b = cv2.inRange(hsv,lower_b,upper_b)

skin_w = cv2.bitwise_and(img, img, mask = skinMask_w)


skin_b = cv2.bitwise_and(img, img, mask = skinMask_b)

skin_add = skin_w + skin_b
height = skin_add.shape[0]
width = skin_add.shape[1]

for i in range(0,height):
    for j in range(0,width):
        if((skinMask_w[i][j]==skinMask_b[i][j]) and (skinMask_w[i][j]!=0)):
            skin_add[i][j] = [0,0,255]
            
mat1 = np.hstack([img,skin_w])
mat2 = np.hstack([skin_b,skin_add])
cv2.imshow("four_images", np.vstack([mat1,mat2]))

cv2.waitKey(0)

cv2.destroyAllWindows() 


