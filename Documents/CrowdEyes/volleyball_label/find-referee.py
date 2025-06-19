import cv2
import numpy as np

points = []
current_point = 0
img = None
scale = 2

def mouse_callback(event, x, y, flags, param):
    global points, current_point, img
    
    if event == cv2.EVENT_LBUTTONDOWN and current_point < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        labels = ['左上', '右上', '左下', '右下']
        cv2.putText(img, labels[current_point], (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        current_point += 1
        
        if current_point == 4:
            # 畫框
            for i in range(4):
                cv2.line(img, points[i], points[(i+1)%4], (0, 255, 0), 2)
            
            # 輸出座標
            print("\n1920x1080尺寸下的座標：")
            for i, label in enumerate(['左上', '右上', '左下', '右下']):
                print(f"{label}: {points[i]}")
            
            # 計算原始尺寸座標
            original_points = [(int(p[0]*scale), int(p[1]*scale)) for p in points]
            x_min = min(p[0] for p in original_points)
            x_max = max(p[0] for p in original_points)
            y_min = min(p[1] for p in original_points)
            y_max = max(p[1] for p in original_points)
            
            print(f"\n裁剪區域代碼：")
            print(f"frame = frames[{y_min}:{y_max}, {x_min}:{x_max}, :]")

def select_referee_area(video_path):
    global img
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("無法開啟影片")
        return
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"原始影片尺寸: {original_width}x{original_height}")
    
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影片幀")
        return
    
    img = cv2.resize(frame, (1920, 1080))
    
    cv2.imshow('Select Points', img)
    cv2.setMouseCallback('Select Points', mouse_callback)
    
    while True:
        cv2.imshow('Select Points', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            current_point = 0
            img = cv2.resize(frame, (1920, 1080))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = '/home/nckusoc/Documents/CrowdEyes/volleyball_label/C0032.MP4'
    # 3840x2160 to 1920x1080
    select_referee_area(video_path)



# 範圍大一點的裁判範圍 (C0015.MP4)
# 1920x1080尺寸下的座標：
# 左上: (868, 516)
# 右上: (1002, 527)
# 左下: (868, 602)
# 右下: (1002, 604)

# 裁剪區域代碼：
# frame = frames[518:630, 892:970, :]


# 裁判範圍 (C0026.MP4)
# 1920x1080尺寸下的座標：
# 左上: (800, 518)
# 右上: (950, 522)
# 左下: (800, 607)
# 右下: (950, 613)