from ultralytics import YOLO
from ultralytics import RTDETR
import json, cv2, time, torch

model = RTDETR('rtdetr-x.pt')
cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/volleyball_label/C0032.MP4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Zhongzheng-32.mp4', fourcc, 30.0, (1920, 1080))

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1920, 1080)


# 1920x1080尺寸下的座標：
# 左上: (845, 475)
# 右上: (981, 479)
# 左下: (845, 556)
# 右下: (984, 565)

action = {}
while True:
   idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
   print(idx)
   ret, frames = cap.read()
   if not ret:
       break
       
   frames = cv2.resize(frames, (1920, 1080))
   frame = frames[475:565, 855:985, :]  
   results = model(frame)
   bbox = results[0].boxes.xyxy
   referee = []
   
   for b in bbox:
       referee = b.cpu().detach().numpy().astype(int).tolist()
       cv2.rectangle(frames, (referee[0] + 850, referee[1] + 479), 
                    (referee[2] + 850, referee[3] + 479), (255, 0, 0), 2)
       referee = [referee[0] + 850, referee[1] + 479, 
                 referee[2] + 850, referee[3] + 479]
       break

   action[str(idx)] = referee
   out.write(frames)
   cv2.imshow('Detection', frames)

   if cv2.waitKey(1) & 0xFF == ord("q"):
       break

js = json.dumps(action)
with open('action-32.json', 'w') as file:
   file.write(js)

cap.release()
out.release()
cv2.destroyAllWindows()