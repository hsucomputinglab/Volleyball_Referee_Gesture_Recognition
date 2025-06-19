"""by lyuwenyu
"""
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
import PIL.Image as Image
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from tqdm import trange
import cv2

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

class RefereeActionDetector:
    def __init__(self):
        # Display settings
        self.serve_display_c = 0
        self.score_display_c = 0
        self.DISPLAY_TIME = 45  
        # Sequence lockout
        self.sequence_lockout = 0
        self.LOCKOUT_TIME = 60  
        # State tracking
        self.serve_state = 0  # 0->1->2
        self.score_state = 0  # 0->2->3
        
        # Detection flags
        self.serve_detected = False
        self.score_detected = False
        
        # Info storage
        self.action_info = ""
        
    def update(self, action):
        """更新狀態並返回檢測結果"""
        current_action = int(action[0][0])
        print(f"\n當前動作: {current_action}")
        
        # 更新鎖定計時器
        if self.sequence_lockout > 0:
            self.sequence_lockout -= 1
            return self._get_status_message()
        
        # 檢查動作序列
        self._check_serve_sequence(current_action)
        self._check_score_sequence(current_action)  
        
        # 更新顯示計數器
        self._update_counters()
        return self._get_status_message()
        
    def _check_serve_sequence(self, action):
        """檢查發球序列 (0->1->2)"""
        print(f"發球狀態: {self.serve_state}, 當前動作: {action}")
        
        if action == 0 and self.serve_state == 0:
            self.serve_state = 1
        elif action == 1 and self.serve_state == 1:
            self.serve_state = 2
        elif action == 2:
            if self.serve_state == 2:
                self.serve_detected = True
                self.serve_display_c = self.DISPLAY_TIME
                self.sequence_lockout = self.LOCKOUT_TIME
                self.action_info = "發球偵測"
            self.serve_state = 0
        
    def _check_score_sequence(self, action):
        """檢查得分序列 (0->2->3)"""
        print(f"得分狀態: {self.score_state}, 當前動作: {action}")
        
        if action == 0 and self.score_state == 0:
            self.score_state = 1
        elif action == 2 and self.score_state == 1:
            self.score_state = 2
        elif action == 3 and self.score_state == 2:
            self.score_detected = True
            self.score_display_c = self.DISPLAY_TIME
            self.sequence_lockout = self.LOCKOUT_TIME
            self.action_info = "得分偵測"
            self.score_state = 0 
                
    def _update_counters(self):
        """更新顯示計數器"""
        if self.serve_display_c > 0:
            self.serve_display_c -= 1
            if self.serve_display_c == 0:
                self.serve_detected = False
                if self.action_info == "發球偵測":
                    self.action_info = ""
        
        if self.score_display_c > 0:
            self.score_display_c -= 1
            if self.score_display_c == 0:
                self.score_detected = False
                if self.action_info == "得分偵測":
                    self.action_info = ""
    
    def _get_status_message(self):
        """獲取當前狀態消息"""
        if self.serve_detected:
            return "發球偵測"
        if self.score_detected:
            return "得分偵測"
        return ""


def draw_frame(img, boxes, status_text, detector):
    """Draw all elements on the frame with status in top-right corner"""
    processed_img = img.copy()
    
    # Draw bounding boxes
    for idx, box in enumerate(boxes):
        cv2.rectangle(processed_img, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 0, 0), 2)

    # 只在有狀態文字時繪製狀態區域
    if status_text:
        # 狀態顯示背景
        status_bg_width = 180
        status_bg_height = 80
        right_margin = processed_img.shape[1] - 20
        top_margin = 20
        
        # 外層深色背景
        cv2.rectangle(processed_img,
                     (right_margin - status_bg_width, top_margin),
                     (right_margin, top_margin + status_bg_height),
                     (245, 245, 245),
                     -1)
        
        # 內層稍淺背景
        cv2.rectangle(processed_img,
                     (right_margin - status_bg_width + 2, top_margin + 2),
                     (right_margin - 2, top_margin + status_bg_height - 2),
                     (255, 255, 255),
                     -1)
        
        # 使用中文字體繪製狀態文字
        text_pos_x = right_margin - status_bg_width + 30
        text_pos_y = top_margin + status_bg_height//2 - 8
        processed_img = draw_chinese_text(processed_img, 
                                        status_text,
                                        (text_pos_x, text_pos_y),
                                        font_size=24, 
                                        bold=True)
        print(f"Drew status text: {status_text}")  # 調試輸出
    
    return processed_img


def create_text_area(status_text, clipper, window_width=1280, text_area_height=100):
    """
    Create a separate text area for video segment info with centered positioning
    
    Args:
        status_text (str): 當前偵測狀態
        clipper (VideoSequenceClipper): 影片切割器實例
        window_width (int): 視窗寬度
        text_area_height (int): 文字區域高度
    """
    # 建立底部白色區域
    text_area = np.ones((text_area_height, window_width, 3), dtype=np.uint8) * 245
    
    # 上方分隔線
    cv2.line(text_area,
             (0, 0),
             (window_width, 0),
             (180, 180, 180),  # 淺灰色分隔線
             2)
    
    # 計算中心位置
    center_x = window_width // 2
    
    # 準備顯示資訊
    info_texts = []
    
    # 偵測事件資訊
    if status_text:
        info_texts.append(f"目前事件：{status_text}")
    else:
        info_texts.append("等待偵測...")
    
    # 錄製狀態
    if clipper.is_recording:
        info_texts.append(f"錄製中 - 已記錄 {clipper.buffer_frame_count} 幀")
    
    # 儲存狀態
    stats = clipper.get_stats()
    info_texts.append(f"已儲存 {stats['clips_saved']} 支影片 / {stats['total_frames']} 幀")
    
    # 在中心位置繪製所有資訊
    for i, text in enumerate(info_texts):
        # 使用較小的字型大小
        font_size = 22
        
        text_area = draw_chinese_text(text_area,
                                    text,
                                    (center_x - 150, 25 + i * 25),  
                                    font_size=font_size,
                                    bold=False)
    
    return text_area

def draw_chinese_text(image, text, position, font_size=36, bold=True):
    """Draw Chinese text with improved visibility and type handling"""
    font_path = '/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/R-PMingLiU-TW-2.ttf'
    font = ImageFont.truetype(font_path, font_size)
    
    # 確保影像格式為 uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    if bold:
        shadow_color = (0, 0, 0)
        for offset in [(0,1), (1,0), (0,-1), (-1,0)]:
            pos = (position[0] + offset[0], position[1] + offset[1])
            draw.text(pos, text, font=font, fill=shadow_color)
    
    text_color = (0, 51, 153)
    draw.text(position, text, font=font, fill=text_color)
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class VideoSequenceClipper:
    def __init__(self, output_dir="Video_Clips"):
        """
        初始化影片切割器
        
        Args:
            output_dir (str): 輸出目錄名稱
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 影片緩衝設定
        self.frame_buffer = []
        
        # 狀態追踪
        self.is_recording = False
        self.serve_frame_idx = None
        self.buffer_frame_count = 0
        
        # 統計資訊
        self.clips_count = 0
        self.total_frames_saved = 0
        
    def update(self, frame, frame_idx, status, cap):
        """
        更新狀態並處理影片切割
        
        Args:
            frame: 當前影格
            frame_idx: 影格索引
            status: 當前狀態 ("發球偵測" 或 "得分偵測")
            cap: 影片捕捉物件
        """
        current_frame = frame.copy()
        
        # 偵測到發球開始錄製
        if status == "發球偵測" and not self.is_recording:
            self._start_recording(frame_idx)
            self.frame_buffer.append(current_frame)
            self.buffer_frame_count += 1
            print(f"開始錄製回合，發球幀：{frame_idx}")
            
        # 正在錄製中
        elif self.is_recording:
            self.frame_buffer.append(current_frame)
            self.buffer_frame_count += 1
            
            # 偵測到得分，結束並儲存
            if status == "得分偵測":
                self._save_sequence(frame_idx, cap)
                self._reset_state()
                print(f"回合結束，得分幀：{frame_idx}")
    
    def _start_recording(self, frame_idx):
        """開始新的錄製"""
        self.is_recording = True
        self.serve_frame_idx = frame_idx
        self.frame_buffer.clear()
        self.buffer_frame_count = 0
        
    def _save_sequence(self, score_frame_idx, cap):
        """儲存影片序列"""
        if not self.frame_buffer:
            print("警告: 緩衝區為空")
            return
            
        try:
            # 檢查輸出目錄
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            
            # 生成檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Volley_{timestamp}_serve{self.serve_frame_idx}_score{score_frame_idx}.mp4"
            output_path = os.path.join(self.output_dir, filename)
            
            # 使用第一幀的尺寸
            frame = self.frame_buffer[0]
            frame_height, frame_width = frame.shape[:2]
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"儲存影片: {output_path}")
            print(f"尺寸: {frame_width}x{frame_height}, FPS: {fps}")
            print(f"總幀數: {len(self.frame_buffer)}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                raise RuntimeError(f"無法創建影片寫入器: {output_path}")
            
            frames_written = 0
            for frame in self.frame_buffer:
                if frame is not None:
                    out.write(frame)
                    frames_written += 1
            # 更新統計
            self.clips_count += 1
            self.total_frames_saved += frames_written
            
            print(f"完成儲存: {frames_written} 幀")
            print(f"檔案大小: {os.path.getsize(output_path)/1024/1024:.1f}MB")
            
        except Exception as e:
            print(f"儲存影片時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if 'out' in locals() and out is not None:
                out.release()
        
    def _reset_state(self):
        """重置所有狀態"""
        self.is_recording = False
        self.serve_frame_idx = None
        self.frame_buffer.clear()
        self.buffer_frame_count = 0
        
    def get_stats(self):
        """取得統計資訊"""
        return {
            'clips_saved': self.clips_count,
            'total_frames': self.total_frames_saved
        }
        
    def __del__(self):
        """清理資源"""
        cv2.destroyAllWindows()

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

    # 設置轉換和模型
    transform = T.Compose([T.ToImageTensor(), T.ConvertDtype()])
    model = Model().eval().cuda().half()

    # 初始化影片捕捉
    cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Full Match ｜ Slovakia vs. Hungary ｜ CEV U18 Volleyball European Championship 2024 [3A9j3pNUanA].webm')
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60000)

    # 初始化緩衝區
    frames = [torch.zeros((576, 1024, 3)).cuda().half() for _ in range(9)]
    images = [np.zeros((576, 1024, 3)) for _ in range(9)]
    f = [0,2,4,6,8]

    # 初始化檢測器
    detector = RefereeActionDetector()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_videos')
    print(f"設置輸出目錄: {output_dir}")
    clipper = VideoSequenceClipper(output_dir=output_dir)

    # 主處理迴圈
    start = time.time()
    with torch.no_grad():
        for i in trange(100000):
            try:
                # 設置顯示視窗
                if i == 0:
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', 1280, 820)  # 720 + 100 for info area

                # 讀取和處理幀
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame (stream end?). Exiting ...")
                    break

                current_frame = cv2.resize(frame, (1024, 576))
                if current_frame is None:
                    print("Error: Failed to resize frame")
                    continue

                # 更新緩衝區
                images.append(current_frame.copy())
                images.pop(0)
                frames.append(transform(torch.tensor(current_frame).cuda()))
                frames.pop(0)

                # 模型推理
                data = torch.stack([frames[i] for i in f], dim=0).permute(3, 0, 1, 2)[None, ...].half()
                out = model(data)
                torch.cuda.synchronize()

                # 處理檢測結果
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

                status = detector.update(action)

                # 影片切割

                current_frame= int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 
                clipper.update(frame.copy(), current_frame, status, cap)  
                processed_img = draw_frame(current_img, boxes, status, detector)

                if processed_img is not None:

                    display_img = cv2.resize(processed_img, (1280, 720))
                    
                    # 合併主畫面和底部資訊
                    # 在主迴圈中
                    text_area = create_text_area(status, clipper, window_width=1280, text_area_height=100)
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
    parser.add_argument('--resume', '-r', type=str, default='/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/X3D_5c_inference/5c_best_inference.pth')

    args = parser.parse_args()

    main(args)

 
# torchrun main.py -c cfg/models/X3D_4816.yaml 
# python export/Clip.py 


# Clip 1h35mins Inference 124 videos -> inference_serve_score_info.csv
