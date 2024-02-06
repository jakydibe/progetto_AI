import cv2 
import numpy as np
import imutils


path = "immagine.png"
def readImage(img_file_path):
    binary_img = None
    img = cv2.imread(img_file_path,0)
    ret,img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary_img = img
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # for each contour search polygon rectangle
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # 0.05
        # print len(approx)


        (x), (y), (w), (h) = cv2.boundingRect(approx)
        #cv2.rectangle(hsv, (x, y),  int( x + int(w / 4)), int(y + int(h / 4)), (255, 255, 255), 13)
        cv2.imshow('frame', img)
    return binary_img

def showImage(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



showImage(readImage(path))
