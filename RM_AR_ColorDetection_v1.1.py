#refernce: https://steam.oxxostudio.tw/category/python/ai/opencv-color-tracking.html

#introduction: This code uses OpenCV to detect red and green colors in real-time video from the camera. 
#It captures video frames, processes them to identify areas of specified colors, and draws rectangles around detected regions.

import cv2
import time
import numpy as np


red_lower = np.array([30,40,200])   # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
red_upper = np.array([90,100,255])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255)

green_lower = np.array([0, 100, 0])    # Lower bound for green
green_upper = np.array([75, 255, 50]) # Upper bound for green

ore_lower = np.array([0, 0, 30])    # H:0-180, S:0-50（低饱和度）, V:30-150（中等亮度）
ore_upper = np.array([180, 50, 150]) # H:0-180, S:0-50（低饱和度）, V:30-150（中等亮度）

"""
# For bright green
green_lower_bgr = np.array([0, 100, 0])
green_upper_bgr = np.array([100, 255, 100])

# For dark green  
green_lower_bgr = np.array([0, 50, 0])
green_upper_bgr = np.array([80, 255, 80])
"""


# create VideoCapture Odject
cap = cv2.VideoCapture(0)  # use default carmera

# set CAP_PROP_FRAME_WIDTH and HEIGHT
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # set height

# set CAP_PROP_FPS
cap.set(cv2.CAP_PROP_FPS, 30)  # set fps to 30 FPS

ptime = 0
ctime = 0

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()  # Read captured frame
    if not ret:
        print("Cannot receive frame")
        break

    img = cv2.resize(img,(640,360))           # 縮小尺寸，加快處理速度

######################################################################################################
#red color detection   
######################################################################################################
    output = cv2.inRange(img, red_lower, red_upper)   # 取得顏色範圍的顏色
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))  # 設定膨脹與侵蝕的參數
    output = cv2.erode(output, kernel)        # 縮小影像，消除雜訊
    output = cv2.dilate(output, kernel)       # 膨脹影像，還原物體大小  
    red_output = cv2.bitwise_and(img, img, mask = output )  # 套用影像遮罩
    cv2.imshow('red', red_output)

    # cv2.findContours 抓取顏色範圍的輪廓座標
    # cv2.RETR_EXTERNAL 表示取得範圍的外輪廓座標串列，cv2.CHAIN_APPROX_SIMPLE 為取值的演算法
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        color = (0,0,255)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)                      # 取得座標與長寬尺寸
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # 繪製四邊形

#####################################################################################################
#green color detection
#####################################################################################################
    output = cv2.inRange(img, green_lower, green_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output = cv2.erode(output, kernel)
    output = cv2.dilate(output, kernel)
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_output = cv2.bitwise_and(img, img, mask = output )
    cv2.imshow('green', green_output)

    for contour in contours:
        area = cv2.contourArea(contour)
        color = (0,255,0)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

#####################################################################################################
#ore detection
#####################################################################################################
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # 中值滤波去噪

    # 霍夫圆检测
    circles = cv2.HoughCircles(
        gray,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # 两圆最小距离（根据矿石大小调整）
        param1=100,  # Canny边缘高阈值
        param2=30,   # 累加器阈值（值越大，检测越严格）
        minRadius=10,  # 最小半径
        maxRadius=50   # 最大半径
    )

    # 绘制检测到的圆
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 坐标和半径取整
        for circle in circles[0, :]:
            x, y, r = circle[0], circle[1], circle[2]
            # 绘制圆心
            cv2.circle(img, (x, y), 2, (0, 255, 0), 3)
            # 绘制圆轮廓
            cv2.circle(img, (x, y), r, (0, 0, 255), 2)
            # 标记坐标和半径
            cv2.putText(img, f"Ore: ({x},{y}), r={r}", (x-r, y-r-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)





#####################################################################################################
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    cv2.imshow("Video", img)  # Show captured frame


    # click 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release carmera and close the window
cap.release()
cv2.destroyAllWindows()
