import cv2
import numpy as np
import time



red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

green_lower = np.array([40, 40, 40])
green_upper = np.array([70, 255, 255])

ore_lower = np.array([0, 0, 50])
ore_upper = np.array([180, 50, 150])

def detect_and_mark(mask, color, img):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{color[0]}_{color[1]}_{color[2]}", (x, y-10), 
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
    ore_mask = cv2.inRange(hsv, ore_lower, ore_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    red_mask = cv2.dilate(red_mask, kernel)
    red_mask = cv2.erode(red_mask, kernel)

    green_mask = cv2.dilate(green_mask, kernel)
    green_mask = cv2.erode(green_mask, kernel)

    ore_mask = cv2.dilate(ore_mask, kernel)
    ore_mask = cv2.erode(ore_mask, kernel)


    # 标记红色板
    img = detect_and_mark(red_mask, (0, 0, 255), img)
    # 标记绿色板
    img = detect_and_mark(green_mask, (0, 255, 0), img)
    # 标记矿石
    contours, _ = cv2.findContours(ore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ore_positions = []  # 存储矿石位置 (x, y, w, h)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # 过滤过小的噪声
            continue
        
        # 计算轮廓的圆度（矿石可能偏圆形）
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.7:  # 过滤过于不规则的轮廓（可调整阈值）
            continue
        
        # 获取边界框坐标
        x, y, w, h = cv2.boundingRect(cnt)
        ore_positions.append((x, y, w, h))
        
        # 在图像上标记矿石
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)  # 橙色框
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
