# -*- coding: utf-8 -*-
import cv2
import numpy as np

cap = cv2.VideoCapture(-1)

def ROI_img(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255, 255, 255])
    roi = cv2.bitwise_and(img, mask)
    return roi

def hough_line(img, rho, theta, threshold, min_line_len, max_line_gap):
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    background_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if(np.all(hough_lines) == True):
        draw_hough_lines(background_img, hough_lines)
    return background_img

def draw_hough_lines(img, lines, color=[29,219,22], thickness = 2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            #cv2.putText(frame,("(%d, %d)" % (x1, y1)),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
            #cv2.putText(frame,("(%d, %d)" % (x2, y2)),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)

def weighted_img(init_img, added_img):
    return cv2.addWeighted(init_img, 1.0, added_img, 1.0, 0.0)



while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.flip(frame,0)

    if(ret):
        height, width = frame.shape[:2]
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.Canny(img, 70, 210)
        
        vertices = np.array([[(150, height/2 - 50), (450, height/2 - 50), (300, height/2 + 50), (100, height/2 + 50)]], np.int32)
        img = ROI_img(img, vertices)
        
        hough_lines = hough_line(img, 1, np.pi/180, 50, 50, 30)
        
        merged_img = weighted_img(frame, hough_lines)
        cv2.imshow('merged_img', merged_img)
        
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()