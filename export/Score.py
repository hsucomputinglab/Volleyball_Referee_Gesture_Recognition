import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import numpy as np
from src.core.yaml_config import YAMLConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import cv2
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

class RefereeActionDetector:
    """Handles referee action detection and state management"""
    def __init__(self):
        # Scoring
        self.left_score = 17
        self.right_score = 20
        
        # State tracking
        self.left_serve_state = 0
        self.right_serve_state = 0
        self.left_win_state = 0
        self.right_win_state = 0
        
        # Display counters
        self.left_win_display_c = 0
        self.right_win_display_c = 0
        self.SCORE_DISPLAY_TIME = 90

        # Score detection cooldown
        self.score_cooldown = 0
        self.SCORE_COOLDOWN_TIME = 120 

        # Complete score sequence lockout
        self.score_sequence_lockout = 0
        self.SCORE_LOCKOUT_TIME = 180  # 更長的序列鎖定時間
       
        # Flags
        self.serve_detected = False
        self.score_detected = False

        # Serve information
        self.serve_info = ""
        self.serve_info_display_count = 60

        # 日志相关
        self.logged_this_event = False
        self.log_path = 'score_log.csv'
        # 如果不存在就创建并写入表头 + 初始比分
        with open(self.log_path, 'w') as f:
            f.write('Time,SVK,HUN\n')
            f.write(f"{datetime.now().strftime('%H:%M:%S')},17,20\n")

    def update(self, action):
        """Update states based on detected action"""
        current_action = action[0][0]
        
        # 更新鎖定計時器
        if self.score_sequence_lockout > 0:
            self.score_sequence_lockout -= 1
            # 在鎖定期間重置所有得分相關狀態
            self.left_win_state = 0
            self.right_win_state = 0
            
        # Process serve detection
        self._check_serve_sequence(current_action)
        
        # 只在沒有鎖定時檢查得分
        if self.score_sequence_lockout == 0:
            if not self.serve_detected:
                self._check_win_sequence(current_action)
            
        # Get status message and update counters
        status_text = self._get_status_message()
        self._update_counters()

        # —— 新增：如果本次循环中刚刚检测到得分，就写一次日志 —— 
        if self.score_detected and not self.logged_this_event:
             with open(self.log_path, 'a') as f:
                 ts = datetime.now().strftime('%H:%M:%S')
                 f.write(f"{ts},{self.left_score},{self.right_score}\n")  # Add
             self.logged_this_event = True
        
        return status_text
        
    def _check_serve_sequence(self, action):
        """Check for serve sequences"""
        # Left Team Serve (0 -> 2)
        if self.left_serve_state == 0 and action == 0:
            self.left_serve_state = 1
            self.serve_info_display_count = 30
        elif self.left_serve_state == 1 and action == 2:
            self.serve_detected = True
            self.serve_info = "左邊隊伍發球"
            self.serve_info_display_count = 30
            self.left_serve_state = 0
        elif action not in [0, 2]:
            self.left_serve_state = 0

        # Right Team Serve (1 -> 3)
        if self.right_serve_state == 0 and action == 1:
            self.right_serve_state = 1
            self.serve_info_display_count = 30
        elif self.right_serve_state == 1 and action == 3:
            self.serve_detected = True
            self.serve_info = "右邊隊伍發球"
            self.serve_info_display_count = 30
            self.right_serve_state = 0
        elif action not in [1, 3]:
            self.right_serve_state = 0

    def _check_win_sequence(self, action):
        """Check for win sequences with strict control"""
        # 如果在顯示時間或冷卻時間內，直接返回
        if self.score_cooldown > 0 or self.left_win_display_c > 0 or self.right_win_display_c > 0:
            return

        # Left Team Win (0 -> 4 -> 5)
        if self.left_win_state == 0 and action == 0:
            self.left_win_state = 1
        elif self.left_win_state == 1 and action == 4:
            self.left_win_state = 2
        elif self.left_win_state == 2 and action == 5:
            if not self.score_detected:
                # 累加左隊分數
                self.left_score += 1
                # 若滿 25 分，則重置雙方分數
                if self.left_score > 25:
                    self.left_score = 0
                    self.right_score = 0

                self.left_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                self.score_detected = True
                # 設置序列鎖定
                self.score_sequence_lockout = self.SCORE_LOCKOUT_TIME
            self.left_win_state = 0
        elif action not in [0, 4, 5]:
            self.left_win_state = 0

        # Right Team Win (1 -> 4 -> 5)
        if self.right_win_state == 0 and action == 1:
            self.right_win_state = 1
        elif self.right_win_state == 1 and action == 4:
            self.right_win_state = 2
        elif self.right_win_state == 2 and action == 5:
            if not self.score_detected:
                # 累加右隊分數
                self.right_score += 1
                # 若滿 25 分，則重置雙方分數
                if self.right_score > 25:
                    self.left_score = 0
                    self.right_score = 0

                self.right_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                self.score_detected = True
                # 設置序列鎖定
                self.score_sequence_lockout = self.SCORE_LOCKOUT_TIME
            self.right_win_state = 0
        elif action not in [1, 4, 5]:
            self.right_win_state = 0

    def _update_counters(self):
        """Update display counters"""
        if self.left_win_display_c > 0:
            self.left_win_display_c -= 1
        if self.right_win_display_c > 0:
            self.right_win_display_c -= 1

        if self.serve_info_display_count > 0:
            self.serve_info_display_count -= 1
        else:
            self.serve_info = ""
            self.serve_detected = False


        if self.score_cooldown > 0:
            self.score_cooldown -= 1
            if self.score_cooldown == 0:
                self.score_detected = False  
                self.logged_this_event = False  # Reset flag

    def _get_status_message(self):
        """Get current status message"""
        if self.left_win_display_c > 0:
            return "左邊隊伍得分"
        if self.right_win_display_c > 0:
            return "右邊隊伍得分"
        if self.serve_info and self.serve_info_display_count > 0:
            return self.serve_info
        return ""


def draw_chinese_text(image, text, position, font_size=36, bold=True):
    """Draw Chinese text on image with bold option"""
    font_path = '/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/R-PMingLiU-TW-2.ttf'
    font = ImageFont.truetype(font_path, font_size)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    # Draw shadow
    if bold:
        for offset in [(0,1), (1,0), (0,-1), (-1,0)]:
            pos = (position[0] + offset[0], position[1] + offset[1])
            draw.text(pos, text, font=font, fill=(0, 0, 0))
    
    draw.text(position, text, font=font, fill=(0, 51, 153))  # deep blue
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_frame(img, boxes, status_text, scores, detector):
    """Draw all elements on the frame"""
    processed_img = img.copy()
    
    # Draw bounding boxes
    for idx, box in enumerate(boxes):
        cv2.rectangle(processed_img, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 255, 255), 2)

    # Score display settings
    score_bg_width = 200
    score_bg_height = 100
    left_margin = 30
    top_margin = 40
    line_spacing = 45

    # Create overlay for semi-transparent background
    overlay = processed_img.copy()
    
    # Outer background with semi-transparency
    cv2.rectangle(overlay, 
                 (left_margin-15, top_margin-35),
                 (left_margin+score_bg_width, top_margin+score_bg_height),
                 (20, 20, 20),  # darker gray
                 -1)
    
    # Inner background with semi-transparency
    cv2.rectangle(overlay, 
                 (left_margin-10, top_margin-30),
                 (left_margin+score_bg_width-5, top_margin+score_bg_height-5),
                 (40, 40, 40),  
                 -1)
    
    # Apply the overlay with transparency
    alpha = 0.9  # 透明度 (0.0 完全透明 到 1.0 完全不透明)
    cv2.addWeighted(overlay, alpha, processed_img, 1 - alpha, 0, processed_img)

    # Horizontal divider line
    line_y = top_margin + line_spacing - 5
    cv2.line(processed_img,
             (left_margin-10, line_y),
             (left_margin+score_bg_width-5, line_y),
             (150, 150, 150), 2)


    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    team_x = left_margin + 10
    score_x = left_margin + 120

    # SVK (Red team) - Using brighter red
    cv2.putText(processed_img, "SVK", 
                (team_x, top_margin + 15), 
                font, 1.0, (0, 50, 255), 2)  # Pure red
    cv2.putText(processed_img, str(detector.left_score),
                (score_x, top_margin + 15),
                font, 1.0, (50, 50, 255), 3)  # Brighter red

    # ITA (Blue team) - Using brighter blue
    cv2.putText(processed_img, "HUN",
                (team_x, top_margin + line_spacing + 35),  # Adjusted Y position
                font, 1.0, (255, 100, 0), 2)  # Pure blue
    cv2.putText(processed_img, str(detector.right_score),
                (score_x, top_margin + line_spacing + 35),  # Adjusted Y position
                font, 1.0, (255, 150, 0), 3)  # Brighter blue
    
    return processed_img

# Draw status message
def create_text_area(status_text, window_width=1280, text_area_height=50):
    """Create a separate text area for status messages"""
    # text_area = np.ones((text_area_height, window_width, 3), dtype=np.uint8) * 255
    text_area = np.ones((text_area_height, window_width, 3), dtype=np.uint8) * 245  # 
    text_font_size = 36
    
    if status_text:
        center_x = window_width // 2 - (len(status_text) * text_font_size // 2.5)
        center_y = text_area_height // 2 - 5 

        text_area = draw_chinese_text(text_area, status_text, 
                                    (center_x, center_y),
                                    font_size=36, bold=True)
    return text_area

def main(args):
    # Load model configuration
    cfg = YAMLConfig(args)
    
    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # Define model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images):
            return self.model(images)

    # Setup transform and model
    transform = T.Compose([T.ToImageTensor(), T.ConvertDtype()])
    model = Model().eval().cuda().half()

    # Initialize video capture
    cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Full Match ｜ Slovakia vs. Hungary ｜ CEV U18 Volleyball European Championship 2024 [3A9j3pNUanA].webm')
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)

    # Initialize frame buffers
    frames = [torch.zeros((576, 1024, 3)).cuda().half() for _ in range(9)]
    images = [np.zeros((576, 1024, 3)) for _ in range(9)]
    f = [0,2,4,6,8]

    # Initialize detector
    detector = RefereeActionDetector()

    # Main processing loop
    start = time.time()
    with torch.no_grad():
        for i in trange(50000):
            try:
                # Setup display window (移到循環外)
                if i == 0:  # 只在第一次創建窗口
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', 1280, 720)

                # Read and process frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame (stream end?). Exiting ...")
                    break

                current_frame = cv2.resize(frame, (1024, 576))
                if current_frame is None:
                    print("Error: Failed to resize frame")
                    continue

                # Update frame buffers
                images.append(current_frame.copy())
                images.pop(0)
                frames.append(transform(torch.tensor(current_frame).cuda()))
                frames.pop(0)

                # Model inference
                data = torch.stack([frames[i] for i in f], dim=0).permute(3, 0, 1, 2)[None, ...].half()
                out = model(data)
                torch.cuda.synchronize()

                # Process detections
                player = out['player']
                current_img = images[4].copy()

                action = player[0][:, :, 4:].cpu().numpy()
                action = np.argmax(action, axis=2)
                
                boxes = player[0][:, :, :4].cpu()
                boxes[:, :, 0] = boxes[:, :, 0] * 1024
                boxes[:, :, 1] = boxes[:, :, 1] * 576
                boxes[:, :, 2] = boxes[:, :, 2] * 1024
                boxes[:, :, 3] = boxes[:, :, 3] * 576
                boxes[:, :, :2], boxes[:, :, 2:] = boxes[:, :, :2] - boxes[:, :, 2:] / 2, boxes[:, :, :2] + boxes[:, :, 2:] / 2
                boxes = boxes[0].cpu().numpy()

                # print('action:', action[0][0])

                # Update detector state and get status
                status_text = detector.update(action)

                # Draw frame
                processed_img = draw_frame(current_img, boxes, "", 
                                        (detector.left_score, detector.right_score), 
                                        detector)

                if processed_img is not None:

                    text_area = create_text_area(status_text, window_width=1280, text_area_height=60)  # 減小高度

                    display_img = cv2.resize(processed_img, (1280, 720))
                    final_display = np.vstack([display_img, text_area])
                    

                    cv2.imshow('frame', final_display)
                else:
                    print("Error: Failed to process frame")
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                continue

    print('FPS:', 1 / ((time.time() - start) / 10000))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', '-c', type=str, default='./cfg/models/X3D_4816.yaml')
    parser.add_argument('--resume', '-r', type=str, default='/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/6c_best_inference.pth')
    args = parser.parse_args()
    main(args)


# python export/Score.py 