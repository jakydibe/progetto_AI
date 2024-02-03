import cv2
import numpy as np
import math




#img = cv2.GaussianBlur(img, (5, 5), 0)
def riconosci_caselle(img,num_immagine):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape), np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
    contour,hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 3)

    res = cv2.bitwise_and(res, mask)

    cv2.imshow("res", res)
    cv2.imshow("mask", mask)
    cv2.waitKey()

    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

    dx = cv2.Sobel(res,cv2.CV_16S,1,0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    cv2.imshow("close", close)
    cv2.waitKey()

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()



    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    dy = cv2.Sobel(res,cv2.CV_16S,0,2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("close-hoizontal", close)
    cv2.waitKey()

    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w/h > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)

    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
    closey = close.copy()

    cv2.imshow('closex',closex)
    cv2.imshow('closey',closey)
    cv2.waitKey()

    res = cv2.bitwise_and(closex,closey)

    cv2.imshow('res',res)
    cv2.waitKey()

    contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
        #cv2.circle(img,(x,y),4,(0,255,0),-1)
        centroids.append((x,y))

    cv2.imshow('img',img)
    cv2.waitKey()


    num_centroids = len(centroids)

    num_righe = int(math.sqrt(num_centroids)) #trovo numero di righe e colonne
    num_colonne = num_righe

    print(num_righe, num_colonne)
        
            

    centroids_sorted_y = sorted(centroids, key=lambda x: x[1])

    for i in range(len(centroids_sorted_y)):
        print(centroids_sorted_y[i], end=" ")

    print("###############################\n")

    array_centr_sorted = []

    for i in range(num_righe):
        start_index = i * num_righe
        end_index = start_index + num_righe
        #if i == num_righe - 1:
        #    end_index = num_righe
        sub_array = centroids_sorted_y[start_index:end_index]
        array_centr_sorted.append(sorted(sub_array, key=lambda x: x[0]))

    for i in range(num_righe):
        for j in range(num_colonne):
            print(array_centr_sorted[i][j], end=" ")

    for i in range(num_righe):
        for j in range(num_colonne):
            x,y = array_centr_sorted[i][j]

            if j == num_colonne - 1 or i == num_righe - 1:
                continue
            else:
                x1 = array_centr_sorted[i][j+1][0]
                y1 = array_centr_sorted[i+1][j][1]

            #formattare in 28x28
            # while(x1 - x > 28):
            #     x1 -= 1
            #     if(x1 - x > 28):
            #         x += 1

            # while(y1 - y > 28):
            #     y1 -= 1
            #     if(y1 - y > 28):
            #         y += 1


            sotto_immagine = img[y:y1, x:x1]
            sotto_immagine = 255 - sotto_immagine
            sotto_immagine = cv2.resize(sotto_immagine, (28,28), interpolation = cv2.INTER_AREA)
            cv2.imwrite(f"C:\\Users\\jakyd\\Desktop\\progetto_AI\\immagini\\sotto_immagine_{num_immagine}_{i}_{j}.png", sotto_immagine)


for i in range(4):
    img = cv2.imread(f'C:\\Users\\jakyd\\Desktop\\progetto_AI\\immagini_labirinti\\immagine_{i}.png')
    riconosci_caselle(img,i)

#cv2.imshow('sotto_immagine',sotto_immagine)
#cv2.waitKey()
