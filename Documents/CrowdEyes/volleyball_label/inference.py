from ultralytics import YOLO
from ultralytics import RTDETR
import json
import cv2
import time
import torch
import matplotlib.pyplot as plt

# Load a model
model = RTDETR('rtdetr-x.pt')

cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/volleyball_label/referee.mkv')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度

fourcc = cv2.VideoWriter_fourcc(*'mp4v')          # 設定影片的格式為 MJPG
out = cv2.VideoWriter('Zhongzheng-15.mp4', fourcc, 30.0, (width,  height))  # 產生空的影片
# cap.set(cv2.CAP_PROP_POS_FRAMES, 13670)
action = {}
while True:
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(idx)
    ret, frames = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    frame = frames[430:730, 810:1110, :]
    results = model(frame)
    bbox = results[0].boxes.xyxy
    referee = []
    for b in bbox:
        if b[1] > 45 and b[3] < 165 and b[0] > 80 and b[2] < 200:
            referee = b.cpu().detach().numpy().astype(int).tolist()
            cv2.rectangle(frames, (referee[0] + 810, referee[1] + 430), (referee[2] + 810, 157 + 430), (255, 0, 0), 2)
            referee = [referee[0] + 810, referee[1] + 430, referee[2] + 810, 157 + 430]
            break

    action[str(idx)] = referee
    # cv2.imshow('test', frames)
    out.write(frames)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

js = json.dumps(action)
file = open('action.json', 'w')
file.write(js)
file.close()

cap.release()
out.release()
cv2.destroyAllWindows()