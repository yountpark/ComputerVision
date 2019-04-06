import cv2
import numpy as np
import matplotlib.pyplot as plt

def error(img, blur):
    w,h,c = img.shape
    img_t = img.astype(np.float)
    blur_t = blur.astype(np.float)
    pixel = w*h*c
    err_mse = abs((img_t - blur_t)**2)
    err_avg = abs(img_t - blur_t)
    return np.sum(err_mse)/pixel ,  np.sum(err_avg)/pixel

def main():
    noise_img = cv2.imread("/Users/yount/Desktop/noise_lena.png", 1)
    origin_img = cv2.imread("/Users/yount/Desktop/lena.png", 1)

    x = (int)(input('x&y :'))
    y = x
    mode = input("mode입력: ")
    if mode == '0':
        blur = cv2.blur(noise_img, (x,y))
        mse, avg = error(origin_img, blur)
        print("mse: %.2f" % mse)
        print("avg: %.2f" % avg)
        cv2.imshow("Blur", blur)

    elif mode == '1':
        blur = cv2.GaussianBlur(noise_img, (x,y), 0)
        mse, avg = error(origin_img, blur)
        print("mse: %.2f" % mse)
        print("avg: %.2f" % avg)
        cv2.imshow("GaussianBlur", blur)

    elif mode == '2':
        blur = cv2.medianBlur(noise_img, x)
        mse, avg = error(origin_img, blur)
        print("mse: %.2f" % mse)
        print("avg: %.2f" % avg)
        cv2.imshow("MedianBlur", blur)
        
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()