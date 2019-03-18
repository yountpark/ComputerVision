# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
def main():
    img = cv2.imread('/Users/yount/Desktop/dataset/4.1.01.tiff', 1)
    print(img.shape)
    width, height, channel = img.shape
    #new = np.zeros_like(img, dtype = np.uint16)

    block_w = int(input("가로 블록 수: "))
    block_h = int(input("세로 블록 수: "))
    block_m = input("블록 모드: ") #median,mean,center
   
    if(block_m == "median"):
        for i in range(0, height, block_h):
            save_h = block_h+i
            for j in range(0, width, block_w):
                save_w = block_w+j
                block_r = []
                block_g = []
                block_b = []
                if(save_h>height):
                    save_h = height
                if(save_w>width):
                    save_w = width
                for k in range(i, save_h):
                    for h in range(j, save_w):
                        block_r.append(img[k][h][0])
                        block_g.append(img[k][h][1])
                        block_b.append(img[k][h][2])
                block_rmedian = int(np.median(block_r))
                block_gmedian = int(np.median(block_g))
                block_bmedian = int(np.median(block_b))
                for k in range(i, save_h):
                    for h in range(j, save_w):
                        img[k][h][0] = block_rmedian
                        img[k][h][1] = block_gmedian
                        img[k][h][2] = block_bmedian
                    
    elif(block_m == "mean"):
        for i in range(0, height, block_h):
            save_h = block_h+i
            for j in range(0, width, block_w):
                save_w = block_w+j
                block_r = []
                block_g = []
                block_b = []
                if(save_h>height):
                    save_h = height
                if(save_w>width):
                    save_w = width
                for k in range(i, save_h):
                    for h in range(j, save_w):
                        block_r.append(img[k][h][0])
                        block_g.append(img[k][h][1])
                        block_b.append(img[k][h][2])
                block_rmean = int(np.mean(block_r))
                block_gmean = int(np.mean(block_g))
                block_bmean = int(np.mean(block_b))
                for k in range(i, save_h):
                    for h in range(j, save_w):
                        img[k][h][0] = block_rmean
                        img[k][h][1] = block_gmean
                        img[k][h][2] = block_bmean
                    
    
    elif(block_m == "center"):
            for i in range(0, height , block_h):
                save_h=block_h+i
                for j in range(0, width , block_w):
                    save_w=block_w+j
                    if(save_h>height):
                        save_h = height
                    if(save_w>width):
                        save_w = width
                    center = img[i+int(block_h/2)][j+int(block_w/2)]
                    for k in range(i,save_h):
                        for h in range(j, save_w):
                            img[k][h]=center
                                    
        
        
        
    
    cv2.imshow('mosaic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()

