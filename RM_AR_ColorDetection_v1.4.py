# RM_colorDetect_v1.4

# description:
# use HSV color space to detect red, green, blue color and ore
# have team mode selection for red team and blue team

#H 通道（色相）：仅表示颜色种类（如红、绿、蓝），与亮度无关。例如，无论红色物体是亮还是暗，其 H 值基本稳定在 0-10 或 170-179 范围内。
#V 通道（明度）：仅表示亮度，与颜色无关。光照变化主要影响这个通道（如强光会使 V 值升高，阴影会使 V 值降低）。
#S 通道（饱和度）：表示颜色的纯净度（0 为灰色，255 为纯彩色），可辅助过滤掉接近白色 / 灰色的 “淡色” 干扰。

import cv2
import numpy as np
import time

Team_Mode = 'z'

red_lower1 = np.array([0, 130, 0])    
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 150, 0])
red_upper2 = np.array([180, 255, 255])

green_lower = np.array([35, 40, 30])
green_upper = np.array([85, 255, 200])

blue_lower = np.array([100, 80, 0])
blue_upper = np.array([130, 255, 255])

ore_lower = np.array([0, 0, 40])
ore_upper = np.array([180, 50, 130])



def input_team_mode():
    team = input("Red Team Mode OR Blue Team Mode (input 'r' or 'b'):")
    return team

def detect_and_mark_color_plate(mask, mark_color, img, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_plate_positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            color_plate_positions.append(((x+x+w)/2, (y+y+h)/2))

            cv2.rectangle(img, (x, y), (x+w, y+h), mark_color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mark_color, 1)
    return img, color_plate_positions

def detect_and_mark_ore(ore_mask, img):
    contours, _ = cv2.findContours(ore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ore_positions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.6:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        ore_positions.append(((x+x+w)/2, (y+y+h)/2))
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(img, "Ore", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    return img, ore_positions


# create VideoCapture Odject
cap = cv2.VideoCapture(0)  # use default carmera

# set CAP_PROP_FRAME_WIDTH and HEIGHT
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # set height

# set CAP_PROP_FPS
cap.set(cv2.CAP_PROP_FPS, 30)  # set fps to 30 FPS

ptime = 0
ctime = 0
counter = 0

Team_Mode = input_team_mode()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

if Team_Mode == 'r':
    while True:
        ret, img = cap.read()
        
        if not ret:
            print("Cannot receive frame")
            break
        
        img = cv2.resize(img,(640,360))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # binary image mask (255 / 0)
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        ore_mask = cv2.inRange(hsv, ore_lower, ore_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        ore_mask = cv2.morphologyEx(ore_mask, cv2.MORPH_CLOSE, kernel)
        ore_mask = cv2.morphologyEx(ore_mask, cv2.MORPH_OPEN, kernel)

        img, red_x = detect_and_mark_color_plate(red_mask, (0,0,255), img, "red")
        img, green_x = detect_and_mark_color_plate(green_mask, (0,255,0), img, "green")
        img, ore_x = detect_and_mark_ore(ore_mask, img)

        red_output = cv2.bitwise_and(img, img, mask = red_mask)
        cv2.imshow("red", red_output)
        green_output = cv2.bitwise_and(img, img, mask = green_mask)
        cv2.imshow("green", green_output)
        ore_output = cv2.bitwise_and(img, img, mask = ore_mask)
        cv2.imshow("ore", ore_output)

        counter = counter + 1
        if(counter == 100):
            print(red_x)
            print(green_x)
            print(ore_x)
            print("=====")
            counter = 0

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (177,197,167), 3)
        cv2.imshow("Video", img)

        # click 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release carmera and close the window
    cap.release()
    cv2.destroyAllWindows()
elif Team_Mode == 'b':
    while True:
        ret, img = cap.read()  # Read captured frame
        
        if not ret:
            print("Cannot receive frame")
            break

        img = cv2.resize(img,(640,360))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        ore_mask = cv2.inRange(hsv, ore_lower, ore_upper)

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        ore_mask = cv2.inRange(hsv, ore_lower, ore_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        ore_mask = cv2.morphologyEx(ore_mask, cv2.MORPH_CLOSE, kernel)
        ore_mask = cv2.morphologyEx(ore_mask, cv2.MORPH_OPEN, kernel)

        img, blue_x = detect_and_mark_color_plate(blue_mask, (255,0,0), img, "blue")
        img, green_x = detect_and_mark_color_plate(green_mask, (0,255,0), img, "green")
        img, ore_x = detect_and_mark_ore(ore_mask, img)

        red_output = cv2.bitwise_and(img, img, mask = blue_mask)
        cv2.imshow("blue", red_output)
        green_output = cv2.bitwise_and(img, img, mask = green_mask)
        cv2.imshow("green", green_output)
        ore_output = cv2.bitwise_and(img, img, mask = ore_mask)
        cv2.imshow("ore", ore_output)

        counter = counter + 1
        if(counter == 10):
            print(blue_x)
            print(green_x)
            print(ore_x)
            print("=====")
            counter = 0

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (177,197,167), 3)
        cv2.imshow("Video", img)

        # click 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release carmera and close the window
    cap.release()
    cv2.destroyAllWindows()
else:
    print("error input!!!")
