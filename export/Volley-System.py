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
import csv


class RefereeActionDetector:
    """Handles referee action detection and state management"""
    def __init__(self):
        # Scoring
        self.left_score = 0  # 初始分數 SVK
        self.right_score = 0  # 初始分數 HUN
        
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
        self.SCORE_LOCKOUT_TIME = 180
        
        # Flags
        self.serve_detected = False
        self.score_detected = False
        self.recording = False
        
        # Serve information
        self.serve_info = ""
        self.serve_info_display_count = 60
        
        # 初始化日誌文件
        self._init_score_log()
        
        # Video clipping - 已註解但保留代碼結構
        # self.frame_buffer = []
        self.clips_count = 0
        self.total_frames_saved = 0
        
    def _init_score_log(self):
        """初始化分數日誌文件"""
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Score_Logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(log_dir, f"ScoreLog_{timestamp}.csv")
        
        # 創建CSV文件並寫入標題行
        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'SVK', 'HUN'])
            
            # 寫入初始分數
            current_time = datetime.now().strftime("%H:%M:%S")
            writer.writerow([current_time, self.left_score, self.right_score])
        
        print(f"分數日誌已初始化: {self.log_filename}")
        
    def _log_score_update(self):
        """記錄分數更新到日誌文件"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        with open(self.log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, self.left_score, self.right_score])
            
        print(f"分數更新已記錄: {current_time}, SVK:{self.left_score}, HUN:{self.right_score}")
        
    def update(self, action, frame, frame_idx, cap):
        """Update states based on detected action"""
        current_action = action[0][0]
        
        # 更新鎖定計時器
        if self.score_sequence_lockout > 0:
            self.score_sequence_lockout -= 1
            self.left_win_state = 0
            self.right_win_state = 0
            
        # Process serve detection
        self._check_serve_sequence(current_action)
        
        # 只在沒有鎖定時檢查得分
        if self.score_sequence_lockout == 0:
            if not self.serve_detected:
                self._check_win_sequence(current_action)
        
        # 註解掉錄製相關功能
        # if self.recording:
        #     self.frame_buffer.append(frame.copy())
        
        # Get status message and update counters
        status_text = self._get_status_message()
        self._update_counters()
        
        return status_text
        

    def _check_serve_sequence(self, action):
        """Check for serve sequences"""

        print(f"當前動作: {action}, 左發球狀態: {self.left_serve_state}, 右發球狀態: {self.right_serve_state}")
        
        # Left Team Serve (0 -> 2)
        if action == 0:
            self.left_serve_state = 1
            self.serve_info_display_count = 30
        elif action == 2 and self.left_serve_state == 1:
            self.serve_detected = True
            self.serve_info = "左邊隊伍發球"
            self.serve_info_display_count = 30
            # self._start_recording()  # 註解掉開始錄製
            print("左隊發球確認")
            self.left_serve_state = 0
        elif action == 1 or action == 3:  # 只在看到另一隊的發球動作時重置
            self.left_serve_state = 0

        # Right Team Serve (1 -> 3)
        if action == 1:
            self.right_serve_state = 1
            self.serve_info_display_count = 30
        elif action == 3 and self.right_serve_state == 1:
            self.serve_detected = True
            self.serve_info = "右邊隊伍發球"
            self.serve_info_display_count = 30
            # self._start_recording()  # 註解掉開始錄製
            print("右隊發球確認")
            self.right_serve_state = 0
        elif action == 0 or action == 2:  # 只在看到另一隊的發球動作時重置
            self.right_serve_state = 0

        # 註解掉錄製相關功能
        # if self.recording:
        #     print(f"目前正在錄製，緩衝區大小: {len(self.frame_buffer)}")


    def _check_win_sequence(self, action):
        """Check for win sequences with strict control"""
        if self.score_cooldown > 0 or self.left_win_display_c > 0 or self.right_win_display_c > 0:
            return

        # Left Team Win (0 -> 4 -> 5)
        if self.left_win_state == 0 and action == 0:
            self.left_win_state = 1
        elif self.left_win_state == 1 and action == 4:
            self.left_win_state = 2
        elif self.left_win_state == 2 and action == 5:
            if not self.score_detected:
                self.left_score += 1
                self.left_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                self.score_detected = True
                self.score_sequence_lockout = self.SCORE_LOCKOUT_TIME
                print(f"左隊得分！目前比分 {self.left_score}:{self.right_score}")
                # 記錄分數更新
                self._log_score_update()
                # 註解掉保存錄製
                # if self.recording:
                #     self._save_current_sequence()
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
                self.right_score += 1
                self.right_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                self.score_detected = True
                self.score_sequence_lockout = self.SCORE_LOCKOUT_TIME
                print(f"右隊得分！目前比分 {self.left_score}:{self.right_score}")
                # 記錄分數更新
                self._log_score_update()
                # 註解掉保存錄製
                # if self.recording:
                #     self._save_current_sequence()
            self.right_win_state = 0
        elif action not in [1, 4, 5]:
            self.right_win_state = 0

    # 註解掉錄製相關功能
    # def _start_recording(self):
    #     """Start a new recording"""
    #     print("開始新的錄製")
    #     self.recording = True
    #     self.frame_buffer.clear()
        
    # def _save_current_sequence(self):
    #     """Save the current video sequence"""
    #     if not self.recording or not self.frame_buffer:
    #         print("沒有影片需要儲存或未在錄製狀態")
    #         return
            
    #     try:
    #         output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video_Clips')
    #         os.makedirs(output_dir, exist_ok=True)
            
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"Match_{timestamp}_L{self.left_score}_R{self.right_score}.mp4"
    #         output_path = os.path.join(output_dir, filename)
            
    #         frame = self.frame_buffer[0]
    #         height, width = frame.shape[:2]
    #         fps = 30.0
            
    #         print(f"開始儲存影片: {output_path}")
    #         print(f"緩衝區幀數: {len(self.frame_buffer)}")
            
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
    #         frames_written = 0
    #         for frame in self.frame_buffer:
    #             out.write(frame)
    #             frames_written += 1
                
    #         self.clips_count += 1
    #         self.total_frames_saved += frames_written
            
    #         print(f"影片儲存完成, 共寫入 {frames_written} 幀")
    #         print(f"目前已儲存 {self.clips_count} 個片段，總計 {self.total_frames_saved} 幀")
            
    #         out.release()
    #         self.recording = False
    #         self.frame_buffer.clear()
            
    #     except Exception as e:
    #         print(f"儲存影片時發生錯誤: {str(e)}")
    #         import traceback
    #         traceback.print_exc()

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

    def _get_status_message(self):
        """Get current status message"""
        if self.left_win_display_c > 0:
            return "左邊隊伍得分"
        if self.right_win_display_c > 0:
            return "右邊隊伍得分"
        if self.serve_info and self.serve_info_display_count > 0:
            return self.serve_info
        return ""

    def get_stats(self):
        """Get recording statistics"""
        return {
            'clips_saved': self.clips_count,
            'total_frames': self.total_frames_saved
        }

def draw_frame(img, boxes, status_text, scores, detector):
    """繪製畫面元素"""
    processed_img = img.copy()
    
    # 繪製框框
    for box in boxes:
        cv2.rectangle(processed_img, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 255, 255), 2)

    # 計分板設置
    score_bg_width = 200
    score_bg_height = 100
    left_margin = 30
    top_margin = 40
    line_spacing = 45

    # 建立半透明背景
    overlay = processed_img.copy()
    cv2.rectangle(overlay, 
                 (left_margin-15, top_margin-35),
                 (left_margin+score_bg_width, top_margin+score_bg_height),
                 (20, 20, 20), -1)
    cv2.rectangle(overlay, 
                 (left_margin-10, top_margin-30),
                 (left_margin+score_bg_width-5, top_margin+score_bg_height-5),
                 (40, 40, 40), -1)
    
    cv2.addWeighted(overlay, 0.9, processed_img, 0.1, 0, processed_img)

    # 分隔線
    line_y = top_margin + line_spacing - 5
    cv2.line(processed_img,
             (left_margin-10, line_y),
             (left_margin+score_bg_width-5, line_y),
             (150, 150, 150), 2)

    # 繪製分數
    font = cv2.FONT_HERSHEY_SIMPLEX
    team_x = left_margin + 10
    score_x = left_margin + 120

    # 左隊分數
    cv2.putText(processed_img, "SVK", 
                (team_x, top_margin + 15), 
                font, 1.0, (0, 50, 255), 2)
    cv2.putText(processed_img, str(detector.left_score),
                (score_x, top_margin + 15),
                font, 1.0, (50, 50, 255), 3)

    # 右隊分數
    cv2.putText(processed_img, "HUN",
                (team_x, top_margin + line_spacing + 35),
                font, 1.0, (255, 100, 0), 2)
    cv2.putText(processed_img, str(detector.right_score),
                (score_x, top_margin + line_spacing + 35),
                font, 1.0, (255, 150, 0), 3)
    
    return processed_img

def create_info_area(status_text, detector, window_width=1280, area_height=60):
    """創建資訊顯示區域"""
    info_area = np.ones((area_height, window_width, 3), dtype=np.uint8) * 245
    
    if status_text:
        center_x = window_width // 2 - (len(status_text) * 36 // 2.5)
        center_y = area_height // 2 - 5
        info_area = draw_chinese_text(info_area, status_text, 
                                    (center_x, center_y),
                                    font_size=36, bold=True)
    
    stats = detector.get_stats()
    stats_text = f"已儲存 {stats['clips_saved']} 個片段，共 {stats['total_frames']} 幀"
    info_area = draw_chinese_text(info_area, stats_text,
                                (10, area_height - 25),
                                font_size=20, bold=False)
    
    return info_area

def draw_chinese_text(image, text, position, font_size=36, bold=True):

    font_path = '/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/R-PMingLiU-TW-2.ttf'  
    font = ImageFont.truetype(font_path, font_size)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    if bold:
        for offset in [(0,1), (1,0), (0,-1), (-1,0)]:
            pos = (position[0] + offset[0], position[1] + offset[1])
            draw.text(pos, text, font=font, fill=(0, 0, 0))
    
    draw.text(position, text, font=font, fill=(0, 51, 153))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def main(args):
  
    cfg = YAMLConfig(args)
    
    print(args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images):
            return self.model(images)

    transform = T.Compose([T.ToImageTensor(), T.ConvertDtype()])
    model = Model().eval().cuda().half()

    
    cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Full Match ｜ Slovakia vs. Hungary ｜ CEV U18 Volleyball European Championship 2024 [3A9j3pNUanA].webm')
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1200) # 可以設定幀數


    frames = [torch.zeros((576, 1024, 3)).cuda().half() for _ in range(9)]
    images = [np.zeros((576, 1024, 3)) for _ in range(9)]
    f = [0,2,4,6,8]

  
    detector = RefereeActionDetector()
    # 註解掉視頻輸出目錄設置
    # output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video_Clips')
    # print(f"設置輸出目錄: {output_dir}")

 
    start = time.time()
    with torch.no_grad():
        for i in trange(50000):
            try:
             
                if i == 0:  
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', 1280, 820)  # 720 + 100 for info area

                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame (stream end?). Exiting ...")
                    break

                current_frame = cv2.resize(frame, (1024, 576))
                if current_frame is None:
                    print("Error: Failed to resize frame")
                    continue

                images.append(current_frame.copy())
                images.pop(0)
                frames.append(transform(torch.tensor(current_frame).cuda()))
                frames.pop(0)

                data = torch.stack([frames[i] for i in f], dim=0).permute(3, 0, 1, 2)[None, ...].half()
                out = model(data)
                torch.cuda.synchronize()

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

                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                status = detector.update(action, frame.copy(), current_frame_idx, cap)

               
                processed_img = draw_frame(current_img, boxes, "", 
                                        (detector.left_score, detector.right_score), 
                                        detector)

                if processed_img is not None:
                    display_img = cv2.resize(processed_img, (1280, 720))
                    text_area = create_info_area(status, detector, window_width=1280, area_height=100)
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
    parser.add_argument('--resume', '-r', type=str, default='/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/output/X3D_6class/checkpoint0085.pth')
    args = parser.parse_args()
    main(args)


 
# torchrun main.py -c cfg/models/X3D_4816.yaml 
# python export/Volley-System.py