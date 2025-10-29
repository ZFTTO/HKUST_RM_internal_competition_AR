import cv2
import numpy as np
import time


red_lower1 = np.array([0, 130, 200])    
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 150, 200])
red_upper2 = np.array([180, 255, 255])

green_lower = np.array([35, 40, 30])
green_upper = np.array([85, 255, 180])

blue_lower = np.array([100, 80, 30])
blue_upper = np.array([130, 255, 180])


ore_lower = np.array([0, 0, 40])
ore_upper = np.array([180, 50, 130])

def detect_and_mark(mask, color, img, label=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label_text = label if label else f"{color[0]}_{color[1]}_{color[2]}"
            cv2.putText(img, label_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


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

    img = cv2.resize(img,(640,360))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    ore_mask = cv2.inRange(hsv, ore_lower, ore_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    ore_mask = cv2.morphologyEx(ore_mask, cv2.MORPH_CLOSE, kernel)
    ore_mask = cv2.morphologyEx(ore_mask, cv2.MORPH_OPEN, kernel)

    # Mark detected objects
    img = detect_and_mark(red_mask, (0, 0, 255), img, "Red")
    img = detect_and_mark(green_mask, (0, 255, 0), img, "Green")
    img = detect_and_mark(blue_mask, (255, 0, 0), img, "Blue")

    red_output = cv2.bitwise_and(img, img, mask =  red_mask)  # 套用影像遮罩
    cv2.imshow('red', red_output)
    green_output = cv2.bitwise_and(img, img, mask =  green_mask)  # 套用影像遮罩
    cv2.imshow('green', green_output)
    blue_output = cv2.bitwise_and(img, img, mask =  blue_mask)  # 套用影像遮罩
    cv2.imshow('blue', blue_output)

    # Mark ore with improved detection
    contours, _ = cv2.findContours(ore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ore_positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.6:  # Slightly lowered threshold for ore detection
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        ore_positions.append((x, y, w, h))
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(img, "Ore", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        ore_output = cv2.bitwise_and(img, img, mask =  ore_mask)  # 套用影像遮罩
        cv2.imshow('ore', ore_output)




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
