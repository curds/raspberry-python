# This Python file uses the following encoding: utf-8
# -*- coding: cp949 -*-
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
import RPi.GPIO as GPIO
import time
import sys, tty, termios, os
import math

input_type = 'video' #'video' # 'image'

# cap = cv2.VideoCapture('solidWhiteRight.mp4')
# cap = cv2.VideoCapture('solidYellowLeft.mp4')
cap = cv2.VideoCapture(-1)
fit_result, l_fit_result, r_fit_result, L_lane, R_lane = [], [], [], [], []


PIN = 18
PWMA1 = 6
PWMA2 = 13
PWMB1 = 20
PWMB2 = 21
D1 = 12
D2 = 26

PWM = 50

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN,GPIO.IN,GPIO.PUD_UP)
GPIO.setup(PWMA1,GPIO.OUT)
GPIO.setup(PWMA2,GPIO.OUT)
GPIO.setup(PWMB1,GPIO.OUT)
GPIO.setup(PWMB2,GPIO.OUT)
GPIO.setup(D1,GPIO.OUT)
GPIO.setup(D2,GPIO.OUT)
p1 = GPIO.PWM(D1,500)
p2 = GPIO.PWM(D2,500)
#p1.start(30)
#p2.start(30)

def set_motor(A1,A2,B1,B2):
    GPIO.output(PWMA1,A1)
    GPIO.output(PWMA2,A2)
    GPIO.output(PWMB1,B1)
    GPIO.output(PWMB2,B2)

def forward(l,r):
    p1.start(l)
    p2.start(r)
    set_motor(0,1,0,1)
    #print("forward")

def stop():
    set_motor(0,0,0,0)

def reverse():
    set_motor(0,1,0,1)

def left(l,r):
    p1.start(l)
    p2.start(r)
    set_motor(1,0,1,0)
    print("left")

def right(l,r):
    p1.start(l)
    p2.start(r)
    set_motor(1,0,1,0)
    
def back():
    GPIO.output(PWMA1,0)
    GPIO.output(PWMA2,1)
    GPIO.output(PWMB1,0)
    GPIO.output(PWMB2,1)





# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, ( 960, 540 ))

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_circle(img, lines, color=[0, 0, 255]):
    for line in lines:
        cv2.circle(img, (line[0], line[1]), 2, color, -1)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_arr, lines)
    return lines


def weighted_img(img, initial_img, j=0.8, q=1., k=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * j + img * q + k
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, j, img, q, k)


def Collect_points(lines):
    # reshape [:4] to [:2]
    interp = lines.reshape(lines.shape[0] * 2, 2)
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3] - line[1]) > 5:
            tmp = np.abs(line[3] - line[1])
            a = line[0];
            b = line[1];
            c = line[2];
            d = line[3]
            slope = (line[2] - line[0]) / (line[3] - line[1])
            for m in range(0, tmp, 5):
                if slope > 0:
                    new_point = np.array([[int(a + m * slope), int(b + m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
                elif slope < 0:
                    new_point = np.array([[int(a - m * slope), int(b - m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
    return interp


def get_random_samples(lines):
    one = random.choice(lines)
    two = random.choice(lines)
    if two[0] == one[0]:  # extract again if values are overlapped
        while two[0] == one[0]:
            two = random.choice(lines)
    one, two = one.reshape(1, 2), two.reshape(1, 2)
    three = np.concatenate((one, two), axis=1)
    three = three.squeeze()
    return three


def compute_model_parameter(line):
    # y = mx+n
    m = (line[3] - line[1]) / (line[2] - line[0])
    n = line[1] - m * line[0]
    # ax+by+c = 0
    a, b, c = m, -1, n
    par = np.array([a, b, c])
    return par


def compute_distance(par, point):
    # distance between line & point

    return np.abs(par[0] * point[:, 0] + par[1] * point[:, 1] + par[2]) / np.sqrt(par[0] ** 2 + par[1] ** 2)


def model_verification(par, lines):
    # calculate distance
    distance = compute_distance(par, lines)
    # total sum of distance between random line and sample points
    sum_dist = distance.sum(axis=0)
    # average
    avg_dist = sum_dist / len(lines)

    return avg_dist


def draw_extrapolate_line(img, par, color=(0, 0, 255), thickness=2):
    x1, y1 = int(-par[1] / par[0] * img.shape[0] - par[2] / par[0]), int(img.shape[0])
    x2, y2 = int(-par[1] / par[0] * (img.shape[0] / 2 + 100) - par[2] / par[0]), int(img.shape[0] / 2 + 100)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def get_fitline(img, f_lines):
    rows, cols = img.shape[:2]
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    if (vy * vx + x) > 0.01:
        x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
        x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
        result = [x1, y1, x2, y2]
    else:
        get_fitline()

    return result


def draw_fitline(img, result_l, result_r, color=(29, 219, 22), thickness=10):
    # draw fitting line
    lane = np.zeros_like(img)
    av0 = (result_r[0] + result_l[0])/2
    av1 = (result_r[1] + result_l[1])/2
    av2 = (result_r[2] + result_l[2])/2
    av3 = (result_r[3] + result_l[3])/2
    cv2.line(lane, (int(result_l[0]), int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]), int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    print([int(result_r[0]),int(result_r[1])])
    print([int(result_r[2]),int(result_r[3])])
        
    cv2.line(lane, (av0, av1), (av2, av3), (0, 0, 255), thickness) 
    #HJ's code
    #yas = av2 - av0
    #cv2.putText(frame,("(%d, %d)" % (av0, av1)),(av0,av1),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
    #cv2.putText(frame,("(%d, %d)" % (av2, av3)),(av2,av3),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
    #if yas > 80:
     #   forward(30,10)
      #  print('right')
    #elif yas < -80:
     #   forward(10,30)
      #  print('left')
    #else:
     #   forward(20,18)
      #  print('forward')
    #forward(25,25)
    
    #print("right line")
    #print(int(result_l[0]))
    #print(int(result_l[1]))
    #print(int(result_l[2]))
    #print(int(result_l[3]))
    x_offset = av2 - av0
    y_offset = int(av3)
    angle_to_mid_radian = math.atan((x_offset + 0.00000001)/ y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    stag = angle_to_mid_deg + 90
    print(stag)
    left = (((stag // 9))) + 19
    right = (180 - stag)//9 + 17
    #right = ((((180 - stag) // 9)*1.5) + 15)
    print("left")
    print(left)
    print("right")
    print(right)
    #if stag < 91:
    #    forward(0,0)
    #    time.sleep(1)
        #time.sleep(0.04)
     #   forward(left,right)
     #   time.sleep(1)
      #  print('right')
    #if stag > 90:
    #    forward(0,0)
    #    time.sleep(1)        
    #    forward(left,right)
    #    time.sleep(1)
      #  print('left')

    # add original image & extracted lane lines
    final = weighted_img(lane, img, 1, 0.5)
    return final


def erase_outliers(par, lines):
    # distance between best line and sample points
    distance = compute_distance(par, lines)

    # filtered_dist = distance[distance<15]
    filtered_lines = lines[distance < 13, :]
    return filtered_lines


def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0, 0, 0, 0])

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line


def ransac_line_fitting(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0, 0, 0])
    if len(lines) != 0:
        for i in range(30):
            sample = get_random_samples(lines)
            parameter = compute_model_parameter(sample)
            cost = model_verification(parameter, lines)
            if cost < min:  # update best_line
                min = cost
                best_line = parameter
            if min < 3: break
        # erase outliers based on best line
        filtered_lines = erase_outliers(best_line, lines)
        fit_result = get_fitline(img, filtered_lines)
    else:
        if ((fit_result[3]+0.0000001) - fit_result[1]) / ((fit_result[2]) - fit_result[0]) < 0:
            l_fit_result = fit_result
            return l_fit_result
        else:
            r_fit_result = fit_result
            return r_fit_result

    if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
        l_fit_result = fit_result
        return l_fit_result
    else:
        r_fit_result = fit_result
        return r_fit_result


def detect_lanes_img(img):
    height, width = img.shape[:2]

    # Set ROI
    vertices = np.array(
        #[[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        [[(5, height/2), (width, height/2), (width, height), (5, height)]],
        dtype=np.int32)
    ROI_img = region_of_interest(img, vertices)

    # Convert to grayimage
    # g_img = grayscale(img)

    # Apply gaussian filter
    blur_img = gaussian_blur(ROI_img, 3)

    # Apply Canny edge transform
    canny_img = canny(blur_img, 70, 210)
    # to except contours of ROI image
    vertices2 = np.array(
       # [[(52, height), (width / 2 - 43, height / 2 + 62), (width / 2 + 43, height / 2 + 62), (width - 52, height)]],
        [[(5, height/2), (width, height/2), (width, height), (5, height)]],
        dtype=np.int32)
    canny_img = region_of_interest(canny_img, vertices2)

    # Perform hough transform
    # Get first candidates for real lane lines
    line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)

    # if can't find any lines
    if line_arr is None:
        return img
    # draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)
    # Get slope degree to separate 2 group (+ slope , - slope)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # ignore vertical slope lines
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    # print(line_arr.shape,'  ',L_lines.shape,'  ',R_lines.shape)

    # if can't find any lines
    if L_lines is None and R_lines is None:
        return img

    # interpolation & collecting points for RANSAC
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines)

    # draw_circle(img,L_interp,(255,255,0))
    # draw_circle(img,R_interp,(0,255,255))

    # erase outliers based on best line
    left_fit_line = ransac_line_fitting(img, L_interp)
    right_fit_line = ransac_line_fitting(img, R_interp)

    # smoothing by using previous frames
    L_lane.append(left_fit_line), R_lane.append(right_fit_line)

    if len(L_lane) > 10:
        left_fit_line = smoothing(L_lane, 10)
    if len(R_lane) > 10:
        right_fit_line = smoothing(R_lane, 10)
    final = draw_fitline(img, left_fit_line, right_fit_line)

    return final

if __name__ == '__main__':
    if input_type == 'image':
        frame = cv2.imread('./test_images/solidYellowCurve.jpg')
        if frame.shape[0] != 540:  # resizing for challenge video
            frame = cv2.resize(frame, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        result = detect_lanes_img(frame)

        cv2.imshow('result', result)
        cv2.waitKey(0)

    elif input_type == 'video':
        while (cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            frame = cv2.flip(frame,0)
               
            #if frame.shape[0] != 540:  # resizing for challenge video
           #     frame = cv2.resize(frame, None, fx=640 / 1, fy=480 / 1, interpolation=cv2.INTER_AREA)
            result = detect_lanes_img(frame)
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
           # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


            cv2.imshow('result', result)

            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()