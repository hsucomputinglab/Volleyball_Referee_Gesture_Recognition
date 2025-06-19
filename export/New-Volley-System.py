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
    """處理裁判動作檢測和狀態管理"""
    def __init__(self):
        # 計分
        self.left_score = 0
        self.right_score = 0
        
        # 狀態追蹤
        self.left_serve_state = 0
        self.right_serve_state = 0
        self.left_win_state = 0
        self.right_win_state = 0
        
        # 得分原因
        self.win_reason = ""
        
        # 顯示計數器
        self.left_win_display_c = 0
        self.right_win_display_c = 0
        self.SCORE_DISPLAY_TIME = 90
        
        # 得分檢測冷卻
        self.score_cooldown = 0
        self.SCORE_COOLDOWN_TIME = 120
        
        # 回合狀態
        self.rally_state = "waiting"  # 可能的狀態: "waiting", "serving", "in_play", "finished"
        
        # 發球保護計數器，防止發球後立即檢測得分
        self.serve_protection_count = 0
        self.SERVE_PROTECTION_TIME = 60  # 減少保護時間，讓球權轉移更快
        
        # 發球信息
        self.serve_info = ""
        self.serve_info_display_count = 0
        
        # 視頻剪輯
        self.recording = False
        self.frame_buffer = []
        self.MAX_BUFFER_SIZE = 1800  # 限制緩衝區大小以防止記憶體問題
        self.clips_count = 0
        self.total_frames_saved = 0

        self.pending_save = False
        self.pending_save_countdown = 0
        self.pending_save_side = ""
        
        # 添加缺少的幀索引變數
        self.serve_frame_idx = None
        self.score_frame_idx = None

    def update(self, action, frame, frame_idx, cap):
        """根據檢測到的動作更新狀態"""
        current_action = action[0][0]
        
        # 加強Debug輸出，顯示所有偵測到的動作值
        all_actions = action[0]  # 取得所有偵測到的動作
        print(f"接收到動作: {current_action}")
        print(f"所有動作偵測內容: {all_actions}")
        print(f"當前回合狀態: {self.rally_state}")
        
        # 檢查是否有特殊動作（5=界內球, 6=界外球，7=觸球）
        if 5 in all_actions:
            print(f"注意: 偵測到界內球(5)")
        if 6 in all_actions:
            print(f"注意: 偵測到界外球(6)")
        if 7 in all_actions:
            print(f"注意: 偵測到觸球(7)")
        
        # 更新發球後的保護計數器
        if self.serve_protection_count > 0:
            self.serve_protection_count -= 1
            if self.serve_protection_count == 0:
                print("發球保護結束，現在可以檢測得分")
                # 發球保護結束後，設置回合狀態為進行中
                if self.rally_state == "serving":
                    self.rally_state = "in_play"
                    print("回合狀態變更為: in_play (球權進行中)")
        
        # 根據回合狀態處理動作
        if self.rally_state == "waiting":
            # 只在等待狀態下檢查發球序列
            serve_detected = self._check_serve_sequence(current_action)
            if serve_detected:
                self.left_win_state = 0
                self.right_win_state = 0
                # add
                self.rally_state = "serving"
                self.serve_protection_count = self.SERVE_PROTECTION_TIME
                # 記錄發球幀索引
                self.serve_frame_idx = frame_idx
                print(f"發球檢測成功，設置保護計數器: {self.SERVE_PROTECTION_TIME} 幀")
                print(f"記錄發球幀索引: {self.serve_frame_idx}")
                print("回合狀態變更為: serving (發球中)")
                
        elif self.rally_state == "in_play" and self.score_cooldown == 0:
            # 只在球權進行中且沒有得分冷卻時檢查得分
            score_detected = self._check_win_sequence(current_action, all_actions, frame_idx)
            if score_detected:
                # 記錄得分幀索引
                self.score_frame_idx = frame_idx
                print(f"記錄得分幀索引: {self.score_frame_idx}")
                self.rally_state = "finished"
                print("回合狀態變更為: finished (回合結束)")
        
        if self.rally_state in ["serving", "in_play"]:
            # 處理影格緩衝區
            if len(self.frame_buffer) < self.MAX_BUFFER_SIZE:
                self.frame_buffer.append(frame.copy())
                print(f"錄製中：已記錄 {len(self.frame_buffer)} 幀")
            else:
                print(f"緩衝區已達到最大容量 ({self.MAX_BUFFER_SIZE}), 丟棄最舊的影格")
                self.frame_buffer.pop(0)  # 移除最舊的影格
                self.frame_buffer.append(frame.copy())
        
        # 獲取狀態信息並更新計數器
        status_text = self._get_status_message()
        print(f"狀態訊息：{status_text}")
        self._update_counters()
        
        return status_text
        
    def _check_serve_sequence(self, action):
        """檢查發球序列 - 根據觀眾視角調整
        0, 2: 左邊發球 (原來是右方隊伍)
        1, 3: 右邊發球 (原來是左方隊伍)
        """
        
        print(f"檢查發球序列 - 當前動作: {action}, 左發球狀態: {self.left_serve_state}, 右發球狀態: {self.right_serve_state}")
        
        serve_detected = False
        
        # 左邊發球 (0 -> 2)
        if action == 0:
            self.left_serve_state = 1
            self.right_serve_state = 0  # 重置另一邊的狀態
            self.serve_info = "左方將發球"
            self.serve_info_display_count = 30
            print("左發球狀態1: 裁判左手舉起")
        elif action == 2 and self.left_serve_state == 1:
            serve_detected = True
            self.serve_info = "左邊隊伍發球"
            self.serve_info_display_count = 60
            self._start_recording()  # 開始錄製
            self.left_serve_state = 0
            print("左發球狀態完成: 左方發球")
        elif action == 1 or action == 3:  # 只在看到另一隊的發球動作時重置
            if self.left_serve_state > 0:
                print("重置左方發球狀態")
            self.left_serve_state = 0

        # 右邊發球 (1 -> 3)
        if action == 1:
            self.right_serve_state = 1
            self.left_serve_state = 0  # 重置另一邊的狀態
            self.serve_info = "右方將發球"
            self.serve_info_display_count = 30
            print("右發球狀態1: 裁判右手舉起")
        elif action == 3 and self.right_serve_state == 1:
            serve_detected = True
            self.serve_info = "右邊隊伍發球"
            self.serve_info_display_count = 60
            self._start_recording()  # 開始錄製
            self.right_serve_state = 0
            print("右發球狀態完成: 右方發球")
        elif action == 0 or action == 2:  # 只在看到另一隊的發球動作時重置
            if self.right_serve_state > 0:
                print("重置右方發球狀態")
            self.right_serve_state = 0

        if self.recording:
            print(f"目前正在錄製，緩衝區大小: {len(self.frame_buffer)} 幀")
        
        return serve_detected
    
    # def _check_win_sequence(self, action, all_actions):
    #     """改進的得分序列檢測邏輯
    #     根據觀眾視角調整得分判斷邏輯:
    #     0 -> 4 -> 5/6/7: 左邊隊伍得分
    #     1 -> 4 -> 5/6/7: 右邊隊伍得分
        
    #     其中5=界內球, 6=界外球, 7=觸球
    #     """
    #     # 檢查是否在冷卻期或已有得分顯示中
    #     if self.score_cooldown > 0 or self.left_win_display_c > 0 or self.right_win_display_c > 0:
    #         return False

    #     print(f"檢查得分序列 - 當前動作: {action}, 左得分狀態: {self.left_win_state}, 右得分狀態: {self.right_win_state}")
    #     print(f"所有偵測到的動作: {all_actions}")
        
    #     # 使用變數記錄是否已經有隊伍得分，避免一次更新中多次得分
    #     has_scored = False
    #     score_detected = False
        
    #     # 記錄初始分數，用於驗證
    #     original_left_score = self.left_score
    #     original_right_score = self.right_score
        
    #     # 檢查是否有終結動作（5、6或7）
    #     terminal_actions = [a for a in all_actions if a in [5, 6, 7]]
    #     has_terminal_action = len(terminal_actions) > 0
    #     terminal_action = terminal_actions[0] if has_terminal_action else None
        
    #     # 打印偵測到的終結動作
    #     if has_terminal_action:
    #         terminal_name = "界內球" if terminal_action == 5 else ("界外球" if terminal_action == 6 else "觸球")
    #         print(f"偵測到終結動作: {terminal_action} ({terminal_name})")
        
    #     # ===== 左隊得分序列檢測 =====
    #     if action == 0 and not has_scored:
    #         # 步驟1: 裁判左手舉起
    #         self.left_win_state = 1
    #         self.right_win_state = 0  # 重置右方狀態，確保互斥
    #         print("左得分序列狀態1: 裁判左手舉起")
        
    #     elif (action == 4 and self.left_win_state == 1) and not has_scored:
    #         # 步驟2: 中間動作
    #         self.left_win_state = 2
    #         print("左得分序列狀態2: 中間動作")
        
    #     elif (action in [5, 6, 7] and self.left_win_state == 1) and not has_scored:
    #         # 左隊得分 - 跳過中間動作直接到終結
    #         print("左隊得分序列: 跳過中間動作，直接到終結")
            
    #         # 設置得分原因 - 明確使用當前動作值
    #         if action == 5:
    #             self.win_reason = "界內球"
    #         elif action == 6:
    #             self.win_reason = "界外球"
    #         elif action == 7:
    #             self.win_reason = "觸球"
            
    #         # 更新分數和狀態
    #         self.left_score += 1
    #         self.left_win_display_c = self.SCORE_DISPLAY_TIME
    #         self.score_cooldown = self.SCORE_COOLDOWN_TIME
    #         has_scored = True
    #         score_detected = True
            
    #         print(f"左隊得分! 得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
            
    #         # 設置延遲保存視頻
    #         self.pending_save = True
    #         self.pending_save_countdown = 30
    #         self.pending_save_side = "left"
            
    #         # 重置狀態
    #         self.left_win_state = 0
    #         self.right_win_state = 0
        
    #     elif ((action in [5, 6, 7] or has_terminal_action) and self.left_win_state == 2) and not has_scored:
    #         # 左隊得分 - 完整序列
    #         # 確定實際終結動作
    #         actual_terminal = action if action in [5, 6, 7] else terminal_action
            
    #         if actual_terminal is not None:
    #             # 設置得分原因 - 使用實際終結動作
    #             if actual_terminal == 5:
    #                 self.win_reason = "界內球"
    #             elif actual_terminal == 6:
    #                 self.win_reason = "界外球"
    #             elif actual_terminal == 7:
    #                 self.win_reason = "觸球"
                
    #             # 更新分數和狀態
    #             self.left_score += 1
    #             self.left_win_display_c = self.SCORE_DISPLAY_TIME
    #             self.score_cooldown = self.SCORE_COOLDOWN_TIME
    #             has_scored = True
    #             score_detected = True
                
    #             print(f"左隊得分! 得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
                
    #             # 設置延遲保存視頻
    #             self.pending_save = True
    #             self.pending_save_countdown = 30
    #             self.pending_save_side = "left"
            
    #         # 重置狀態
    #         self.left_win_state = 0
    #         self.right_win_state = 0
        
    #     # ===== 右隊得分序列檢測 =====
    #     if action == 1 and not has_scored:
    #         # 步驟1: 裁判右手舉起
    #         self.right_win_state = 1
    #         self.left_win_state = 0  # 重置左方狀態，確保互斥
    #         print("右得分序列狀態1: 裁判右手舉起")
        
    #     elif (action == 4 and self.right_win_state == 1) and not has_scored:
    #         # 步驟2: 中間動作
    #         self.right_win_state = 2
    #         print("右得分序列狀態2: 中間動作")
        
    #     elif (action in [5, 6, 7] and self.right_win_state == 1) and not has_scored:
    #         # 右隊得分 - 跳過中間動作直接到終結
    #         print("右隊得分序列: 跳過中間動作，直接到終結")
            
    #         # 設置得分原因 - 明確使用當前動作值
    #         if action == 5:
    #             self.win_reason = "界內球"
    #         elif action == 6:
    #             self.win_reason = "界外球"
    #         elif action == 7:
    #             self.win_reason = "觸球"
            
    #         # 更新分數和狀態
    #         self.right_score += 1
    #         self.right_win_display_c = self.SCORE_DISPLAY_TIME
    #         self.score_cooldown = self.SCORE_COOLDOWN_TIME
    #         has_scored = True
    #         score_detected = True
            
    #         print(f"右隊得分! 得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
            
    #         # 設置延遲保存視頻
    #         self.pending_save = True
    #         self.pending_save_countdown = 30
    #         self.pending_save_side = "right"
            
    #         # 重置狀態
    #         self.right_win_state = 0
    #         self.left_win_state = 0
        
    #     elif ((action in [5, 6, 7] or has_terminal_action) and self.right_win_state == 2) and not has_scored:
    #         # 右隊得分 - 完整序列
    #         # 確定實際終結動作
    #         actual_terminal = action if action in [5, 6, 7] else terminal_action
            
    #         if actual_terminal is not None:
    #             # 設置得分原因 - 使用實際終結動作
    #             if actual_terminal == 5:
    #                 self.win_reason = "界內球"
    #             elif actual_terminal == 6:
    #                 self.win_reason = "界外球"
    #             elif actual_terminal == 7:
    #                 self.win_reason = "觸球"
                
    #             # 更新分數和狀態
    #             self.right_score += 1
    #             self.right_win_display_c = self.SCORE_DISPLAY_TIME
    #             self.score_cooldown = self.SCORE_COOLDOWN_TIME
    #             has_scored = True
    #             score_detected = True
                
    #             print(f"右隊得分! 得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
                
    #             # 設置延遲保存視頻
    #             self.pending_save = True
    #             self.pending_save_countdown = 30
    #             self.pending_save_side = "right"
            
    #         # 重置狀態
    #         self.right_win_state = 0
    #         self.left_win_state = 0
        
    #     # ===== 單幀完整序列檢測 =====
    #     # 只有在之前沒有得分的情況下才檢查
    #     if not has_scored:
    #         # 左方完整序列 (0->4->終結動作)
    #         if 0 in all_actions and 4 in all_actions and any(t in all_actions for t in [5, 6, 7]):
    #             # 找到終結動作（從優先級最高的開始）
    #             for priority_terminal in [7, 6, 5]:  # 優先順序: 觸球 > 界外球 > 界內球
    #                 if priority_terminal in all_actions:
    #                     terminal_act = priority_terminal
    #                     break
    #             else:
    #                 terminal_act = next((t for t in [5, 6, 7] if t in all_actions), None)
                
    #             if terminal_act:
    #                 # 設置得分原因
    #                 if terminal_act == 5:
    #                     self.win_reason = "界內球"
    #                 elif terminal_act == 6:
    #                     self.win_reason = "界外球"
    #                 elif terminal_act == 7:
    #                     self.win_reason = "觸球"
                    
    #                 # 更新分數和狀態
    #                 self.left_score += 1
    #                 self.left_win_display_c = self.SCORE_DISPLAY_TIME
    #                 self.score_cooldown = self.SCORE_COOLDOWN_TIME
    #                 has_scored = True
    #                 score_detected = True
                    
    #                 print(f"左隊得分! (單幀完整序列) 得分原因: {self.win_reason}")
                    
    #                 # 設置延遲保存視頻
    #                 self.pending_save = True
    #                 self.pending_save_countdown = 30
    #                 self.pending_save_side = "left"
                    
    #                 # 重置狀態
    #                 self.left_win_state = 0
    #                 self.right_win_state = 0
            
    #         # 右方完整序列 (1->4->終結動作)
    #         elif 1 in all_actions and 4 in all_actions and any(t in all_actions for t in [5, 6, 7]):
    #             # 找到終結動作（從優先級最高的開始）
    #             for priority_terminal in [7, 6, 5]:  # 優先順序: 觸球 > 界外球 > 界內球
    #                 if priority_terminal in all_actions:
    #                     terminal_act = priority_terminal
    #                     break
    #             else:
    #                 terminal_act = next((t for t in [5, 6, 7] if t in all_actions), None)
                
    #             if terminal_act:
    #                 # 設置得分原因
    #                 if terminal_act == 5:
    #                     self.win_reason = "界內球"
    #                 elif terminal_act == 6:
    #                     self.win_reason = "界外球"
    #                 elif terminal_act == 7:
    #                     self.win_reason = "觸球"
                    
    #                 # 更新分數和狀態
    #                 self.right_score += 1
    #                 self.right_win_display_c = self.SCORE_DISPLAY_TIME
    #                 self.score_cooldown = self.SCORE_COOLDOWN_TIME
    #                 has_scored = True
    #                 score_detected = True
                    
    #                 print(f"右隊得分! (單幀完整序列) 得分原因: {self.win_reason}")
                    
    #                 # 設置延遲保存視頻
    #                 self.pending_save = True
    #                 self.pending_save_countdown = 30
    #                 self.pending_save_side = "right"
                    
    #                 # 重置狀態
    #                 self.right_win_state = 0
    #                 self.left_win_state = 0
        
    #     # ===== 狀態維護 =====
    #     # 處理非序列動作的狀態維護
    #     if action not in [0, 1, 4, 5, 6, 7]:
    #         # 增加非序列動作計數器
    #         if not hasattr(self, 'non_sequence_count'):
    #             self.non_sequence_count = 0
            
    #         self.non_sequence_count += 1
            
    #         # 連續3次檢測到不相關動作才重置狀態，避免臨時干擾
    #         if self.non_sequence_count >= 3:
    #             if self.left_win_state > 0:
    #                 print(f"重置左方得分狀態，因為連續偵測到不相關動作: {action}")
    #                 self.left_win_state = 0
    #             if self.right_win_state > 0:
    #                 print(f"重置右方得分狀態，因為連續偵測到不相關動作: {action}")
    #                 self.right_win_state = 0
    #             self.non_sequence_count = 0
    #     else:
    #         # 重置計數器
    #         self.non_sequence_count = 0
        
    #     # 檢查狀態異常 - 確保左右隊的狀態不會同時存在
    #     if self.left_win_state > 0 and self.right_win_state > 0:
    #         print("異常: 左右得分狀態同時存在，根據當前動作決定保留哪一方")
    #         if action in [0, 2, 5, 6, 7] and self.left_win_state > 0:
    #             self.right_win_state = 0
    #         elif action in [1, 3] and self.right_win_state > 0:
    #             self.left_win_state = 0
    #         else:
    #             # 無法判斷時都重置
    #             self.left_win_state = 0
    #             self.right_win_state = 0
        
    #     # 驗證分數變化
    #     if has_scored:
    #         print(f"得分檢測結果: {'成功' if score_detected else '失敗'}, "
    #             f"左隊: {original_left_score}->{self.left_score}, "
    #             f"右隊: {original_right_score}->{self.right_score}")
        
    #     # 檢查finished狀態下的動作
    #     if self.rally_state == "finished" and action in [0, 1]:
    #         print(f"警告: 回合已結束但仍偵測到得分相關動作: {action}")
        
    #     return score_detected

    def _check_win_sequence(self, action, all_actions, current_frame_idx):
        """改進的得分序列檢測邏輯
        根據觀眾視角調整得分判斷邏輯:
        0 -> 4 -> 5/6/7: 左邊隊伍得分
        1 -> 4 -> 5/6/7: 右邊隊伍得分
        
        其中5=界內球, 6=界外球, 7=觸球
        """
        if self.score_cooldown > 0 or self.left_win_display_c > 0 or self.right_win_display_c > 0:
            return False

        print(f"檢查得分序列 - 當前動作: {action}, 左得分狀態: {self.left_win_state}, 右得分狀態: {self.right_win_state}")
        print(f"所有偵測到的動作: {all_actions}")
        
        score_detected = False
        
        # 檢查是否有終結動作（5、6或7）存在於所有檢測到的動作中
        terminal_actions_present = [a for a in all_actions if a in [5, 6, 7]]
        has_terminal_action = len(terminal_actions_present) > 0
        
        # 如果有多個終結動作，優先使用出現在列表前面的（置信度更高的）
        terminal_action = terminal_actions_present[0] if has_terminal_action else None
        
        # 初始化序列開始幀的屬性（如果不存在）
        if not hasattr(self, 'left_sequence_start_frame'):
            self.left_sequence_start_frame = 0
        if not hasattr(self, 'right_sequence_start_frame'):
            self.right_sequence_start_frame = 0
        if not hasattr(self, 'sequence_timeout'):
            self.sequence_timeout = 30  # 預設的序列超時幀數
        
        # 左方隊伍得分序列 (0 -> 4 -> [5/6/7])
        if action == 0:
            # 進入左方得分序列第一步
            self.left_win_state = 1
            self.right_win_state = 0  # 重置右方狀態
            self.left_sequence_start_frame = current_frame_idx  # 記錄序列開始的幀
            print("左得分序列狀態1: 裁判左手舉起")
        
        # 左方得分序列的直接終結動作檢測（跳過中間動作）
        elif action in [5, 6, 7] and self.left_win_state == 1 and (current_frame_idx - self.left_sequence_start_frame) < self.sequence_timeout:
            print(f"左得分序列: 跳過中間動作，直接到終結動作 {action}")
            self.left_score += 1
            self.left_win_display_c = self.SCORE_DISPLAY_TIME
            self.score_cooldown = self.SCORE_COOLDOWN_TIME
            score_detected = True
            
            # 設置得分原因
            if action == 5:
                self.win_reason = "界內球"
                print("偵測到界內球(5)動作")
            elif action == 6:
                self.win_reason = "界外球"
                print("偵測到界外球(6)動作")
            elif action == 7:
                self.win_reason = "觸球"
                print("偵測到觸球(7)動作")
            
            print(f"左隊得分！得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
            
            # 設置延遲保存視頻
            self.pending_save = True
            self.pending_save_countdown = 5
            self.pending_save_side = "left"
            self.left_win_state = 0  # 重置狀態
        
        # 左方得分序列第二步：中間動作4
        elif action == 4 and self.left_win_state == 1:
            # 正常序列：檢測到中間動作
            self.left_win_state = 2
            print("左得分序列狀態2: 中間動作")
        
        # 左方得分序列終結步驟：檢測到終結動作或在所有動作中存在終結動作
        elif (action in [5, 6, 7] or has_terminal_action) and self.left_win_state == 2:
            # 確定實際的終結動作
            actual_terminal_action = action if action in [5, 6, 7] else terminal_action
            
            if actual_terminal_action is not None:
                self.left_score += 1
                self.left_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                score_detected = True
                
                # 設置得分原因
                if actual_terminal_action == 5:
                    self.win_reason = "界內球"
                    print("偵測到界內球(5)動作")
                elif actual_terminal_action == 6:
                    self.win_reason = "界外球"
                    print("偵測到界外球(6)動作")
                elif actual_terminal_action == 7:
                    self.win_reason = "觸球"
                    print("偵測到觸球(7)動作")
                
                print(f"左隊得分！得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
                
                # 設置延遲保存視頻
                self.pending_save = True
                self.pending_save_countdown = 5
                self.pending_save_side = "left"
            
            self.left_win_state = 0  # 重置狀態
        
        # 右方隊伍得分序列 (1 -> 4 -> [5/6/7])
        if action == 1:
            # 進入右方得分序列第一步
            self.right_win_state = 1
            self.left_win_state = 0  # 重置左方狀態
            self.right_sequence_start_frame = current_frame_idx  # 記錄序列開始的幀
            print("右得分序列狀態1: 裁判右手舉起")
        
        # 右方得分序列的直接終結動作檢測（跳過中間動作）
        elif action in [5, 6, 7] and self.right_win_state == 1 and (current_frame_idx - self.right_sequence_start_frame) < self.sequence_timeout:
            print(f"右得分序列: 跳過中間動作，直接到終結動作 {action}")
            self.right_score += 1
            self.right_win_display_c = self.SCORE_DISPLAY_TIME
            self.score_cooldown = self.SCORE_COOLDOWN_TIME
            score_detected = True
            
            # 設置得分原因
            if action == 5:
                self.win_reason = "界內球"
                print("偵測到界內球(5)動作")
            elif action == 6:
                self.win_reason = "界外球"
                print("偵測到界外球(6)動作")
            elif action == 7:
                self.win_reason = "觸球"
                print("偵測到觸球(7)動作")
            
            print(f"右隊得分！得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
            
            # 設置延遲保存視頻
            self.pending_save = True
            self.pending_save_countdown = 5
            self.pending_save_side = "right"
            self.right_win_state = 0  # 重置狀態
        
        # 右方得分序列第二步：中間動作4
        elif action == 4 and self.right_win_state == 1:
            # 正常序列：檢測到中間動作
            self.right_win_state = 2
            print("右得分序列狀態2: 中間動作")
        
        # 右方得分序列終結步驟：檢測到終結動作或在所有動作中存在終結動作
        elif (action in [5, 6, 7] or has_terminal_action) and self.right_win_state == 2:
            # 確定實際的終結動作
            actual_terminal_action = action if action in [5, 6, 7] else terminal_action
            
            if actual_terminal_action is not None:
                self.right_score += 1
                self.right_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                score_detected = True
                
                # 設置得分原因
                if actual_terminal_action == 5:
                    self.win_reason = "界內球"
                    print("偵測到界內球(5)動作")
                elif actual_terminal_action == 6:
                    self.win_reason = "界外球"
                    print("偵測到界外球(6)動作")
                elif actual_terminal_action == 7:
                    self.win_reason = "觸球"
                    print("偵測到觸球(7)動作")
                
                print(f"右隊得分！得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
                
                # 設置延遲保存視頻
                self.pending_save = True
                self.pending_save_countdown = 5
                self.pending_save_side = "right"
            
            self.right_win_state = 0  # 重置狀態
        
        # 新增：直接檢測完整序列
        # 如果在一個短時間窗口內檢測到完整序列但可能因為幀率問題未能捕捉中間狀態
        # 這種情況通常發生在裁判動作快速或模型略過了某些中間狀態
        
        # 左方完整序列檢測
        if 0 in all_actions and 4 in all_actions and any(t in all_actions for t in [5, 6, 7]) and self.left_win_state == 0 and self.right_win_state == 0:
            # 找到終結動作
            terminal_act = next((t for t in [5, 6, 7] if t in all_actions), None)
            if terminal_act:
                print("檢測到左方完整得分序列 (0->4->終結動作) 在單一幀中")
                self.left_score += 1
                self.left_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                score_detected = True
                
                # 設置得分原因
                if terminal_act == 5:
                    self.win_reason = "界內球"
                elif terminal_act == 6:
                    self.win_reason = "界外球"
                elif terminal_act == 7:
                    self.win_reason = "觸球"
                    
                print(f"左隊得分！得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
                
                # 設置延遲保存視頻
                self.pending_save = True
                self.pending_save_countdown = 30
                self.pending_save_side = "left"
        
        # 右方完整序列檢測
        if 1 in all_actions and 4 in all_actions and any(t in all_actions for t in [5, 6, 7]) and self.left_win_state == 0 and self.right_win_state == 0:
            # 找到終結動作
            terminal_act = next((t for t in [5, 6, 7] if t in all_actions), None)
            if terminal_act:
                print("檢測到右方完整得分序列 (1->4->終結動作) 在單一幀中")
                self.right_score += 1
                self.right_win_display_c = self.SCORE_DISPLAY_TIME
                self.score_cooldown = self.SCORE_COOLDOWN_TIME
                score_detected = True
                
                # 設置得分原因
                if terminal_act == 5:
                    self.win_reason = "界內球"
                elif terminal_act == 6:
                    self.win_reason = "界外球"
                elif terminal_act == 7:
                    self.win_reason = "觸球"
                    
                print(f"右隊得分！得分原因: {self.win_reason}，目前比分 {self.left_score}:{self.right_score}")
                
                # 設置延遲保存視頻
                self.pending_save = True
                self.pending_save_countdown = 30
                self.pending_save_side = "right"
        
        # 狀態維護：更寬容的狀態重置策略
        if action not in [0, 1, 4, 5, 6, 7]:
            # 使用計數器確保連續多次檢測到不相關動作才重置狀態
            if not hasattr(self, 'non_sequence_count'):
                self.non_sequence_count = 0
                
            self.non_sequence_count += 1
            
            # 需要連續5次檢測到不相關動作才重置狀態 (增加容忍度)
            if self.non_sequence_count >= 10:
                if self.left_win_state > 0:
                    print(f"重置左方得分狀態，因為連續5次偵測到不相關動作 {action}")
                    self.left_win_state = 0
                if self.right_win_state > 0:
                    print(f"重置右方得分狀態，因為連續5次偵測到不相關動作 {action}")
                    self.right_win_state = 0
                self.non_sequence_count = 0
        else:
            # 重置計數器
            self.non_sequence_count = 0
        
        return score_detected

   
    def _start_recording(self):
        """開始新的錄製"""
        print("開始新的錄製")
        self.recording = True
        self.frame_buffer.clear()
        
    def _save_current_sequence(self):
        """保存當前視頻序列"""
        if not self.frame_buffer:
            print("沒有影片需要儲存或未在錄製狀態")
            return
            
        try:
            import os
            from datetime import datetime
            import cv2
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video_Clips')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 使用安全的方式設置幀索引
            sf = self.serve_frame_idx if self.serve_frame_idx is not None else 0
            sc = self.score_frame_idx if self.score_frame_idx is not None else 0
            reason = self.win_reason or "rally"
            filename = (
                f"Match_{timestamp}"
                f"_serve{sf:06d}"
                f"_score{sc:06d}"
                f"_L{self.left_score}"
                f"_R{self.right_score}"
                f"_{reason}.mp4"
            )
            output_path = os.path.join(output_dir, filename)
            
            if not self.frame_buffer:
                print("警告: 影格緩衝區為空，無法儲存影片")
                return
                
            frame = self.frame_buffer[0]
            if frame is None:
                print("警告: 第一個影格為空，無法儲存影片")
                self.frame_buffer.clear()
                return
            
            # 檢查frame是否為有效的numpy陣列
            if not isinstance(frame, np.ndarray):
                print(f"警告: 影格不是有效的numpy陣列，類型為: {type(frame)}")
                self.frame_buffer.clear()
                return
                
            # 檢查frame的形狀
            if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                print(f"警告: 影格尺寸無效，尺寸為: {frame.shape}")
                self.frame_buffer.clear()
                return
                
            # 固定輸出解析度為 1920x1080
            target_width = 1920
            target_height = 1080
            fps = 30.0
            
            print(f"開始儲存影片: {output_path}")
            print(f"緩衝區幀數: {len(self.frame_buffer)}")
            print(f"影片尺寸: {target_width}x{target_height}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            if not out.isOpened():
                print(f"無法創建視頻寫入器: {output_path}")
                self.frame_buffer.clear()
                return
            
            frames_written = 0
            for frame in self.frame_buffer:
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0 and frame.shape[1] > 0:
                    try:
                        # 確保每個影格都調整到 1920x1080
                        resized_frame = cv2.resize(frame, (target_width, target_height))
                        out.write(resized_frame)
                        frames_written += 1
                    except Exception as e:
                        print(f"寫入影格時發生錯誤: {str(e)}")
                else:
                    print("跳過無效的影格")
                
            self.clips_count += 1
            self.total_frames_saved += frames_written
            
            print(f"影片儲存完成, 共寫入 {frames_written} 幀")
            print(f"目前已儲存 {self.clips_count} 個片段，總計 {self.total_frames_saved} 幀")
            
            out.release()
            self.frame_buffer.clear()
            # 視頻保存完成後，重置回合狀態為等待狀態
            self.rally_state = "waiting"
            print("回合狀態變更為: waiting (等待發球)")
            self.recording = False  # 重置錄製狀態
            
        except Exception as e:
            print(f"儲存影片時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            # 即使出錯也重置狀態以防止卡住
            self.frame_buffer.clear()
            # 重置回合狀態
            self.rally_state = "waiting"
            self.recording = False  # 重置錄製狀態
            print("因錯誤重置回合狀態為: waiting (等待發球)")

    def _update_counters(self):
        """更新顯示計數器"""
        if self.left_win_display_c > 0:
            self.left_win_display_c -= 1
            if self.left_win_display_c == 0:
                print("左隊得分顯示結束")
        
        if self.right_win_display_c > 0:
            self.right_win_display_c -= 1
            if self.right_win_display_c == 0:
                print("右隊得分顯示結束")

        if self.serve_info_display_count > 0:
            self.serve_info_display_count -= 1
            if self.serve_info_display_count == 0:
                print("發球信息顯示結束")
                self.serve_info = ""
        
        if self.score_cooldown > 0:
            self.score_cooldown -= 1
            if self.score_cooldown == 0:
                print("得分冷卻結束")
                
        # 處理延遲保存視頻
        if self.pending_save:
            self.pending_save_countdown -= 1
            print(f"延遲保存倒計時: {self.pending_save_countdown}")
            if self.pending_save_countdown <= 0:
                print(f"狀態顯示完成，現在保存{self.pending_save_side}隊得分視頻...")
                self._save_current_sequence()
                self.pending_save = False

    def _get_status_message(self):
        """獲取當前狀態信息，帶有詳細原因"""
        if self.left_win_display_c > 0:
            return f"左邊隊伍得分，因為{self.win_reason}"
        if self.right_win_display_c > 0:
            return f"右邊隊伍得分，因為{self.win_reason}"
        if self.serve_info and self.serve_info_display_count > 0:
            return self.serve_info
        return ""

    def get_stats(self):
        """獲取錄製統計資料"""
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
   # 建立半透明背景 - 白色系
    overlay = processed_img.copy()
    cv2.rectangle(overlay, 
                (left_margin-15, top_margin-35),
                (left_margin+score_bg_width, top_margin+score_bg_height),
                (240, 240, 240), -1)  # 外層淺灰色
    cv2.rectangle(overlay, 
                (left_margin-10, top_margin-30),
                (left_margin+score_bg_width-5, top_margin+score_bg_height-5),
                (250, 250, 250), -1)  # 內層近乎白色

    # 調整透明度 - 白色背景需較高透明度才能清晰顯示
    cv2.addWeighted(overlay, 0.7, processed_img, 0.3, 0, processed_img)

    # 分隔線
    line_y = top_margin + line_spacing - 5
    cv2.line(processed_img,
             (left_margin-10, line_y),
             (left_margin+score_bg_width-5, line_y),
             (150, 150, 150), 2)

    # 轉換為PIL圖像以繪製中文
    pil_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_path = '/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/R-PMingLiU-TW-2.ttf'
    team_font = ImageFont.truetype(font_path, 24)  # 隊名用較小字體
    score_font = ImageFont.truetype(font_path, 24)  # 分數用較大字體
    
    team_x = left_margin + 10
    score_x = left_margin + 160
    
    # 左隊名稱和分數
    draw.text((team_x, top_margin + line_spacing + 10), "臺北conti", font=team_font, fill=(220, 20, 60))
    draw.text((score_x, top_margin + line_spacing + 10), str(detector.right_score), font=score_font, fill=(255, 0, 0))
    
    
    # 右隊名稱和分數
    draw.text((team_x, top_margin - 10), "雲林美津濃", font=team_font, fill=(0, 128, 255))
    draw.text((score_x, top_margin - 10), str(detector.left_score), font=score_font, fill=(0, 128, 255))

    # 這裏交換以便與電子計分板一致
    # 可以後續再修改

    processed_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
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
    # 設定顯示紀錄更多資訊
    np.set_printoptions(precision=3, suppress=True)
  
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

    cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/C0033.MP4')
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 可以設定幀數


    frames = [torch.zeros((576, 1024, 3)).cuda().half() for _ in range(9)]
    images = [np.zeros((576, 1024, 3)) for _ in range(9)]
    f = [0,2,4,6,8]

  
    detector = RefereeActionDetector()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video_Clips')
    os.makedirs(output_dir, exist_ok=True)
    print(f"設置輸出目錄: {output_dir}")

 
    start = time.time()
    with torch.no_grad():
        for i in trange(50000):
            try:
                if i == 0:  
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', 1280, 720 + 100)  # 1080 + 100 for info area

                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame (stream end?). Exiting ...")
                    break

                original_frame = frame.copy()
                
                # 調整大小進行處理
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
                
                status = detector.update(action, original_frame, current_frame_idx, cap)
                

                processed_img = draw_frame(current_img, boxes, status, 
                                        (detector.left_score, detector.right_score), 
                                        detector)

                if processed_img is not None:
                    display_img = cv2.resize(processed_img, (1024, 576))
                    

                    text_area = create_info_area(status, detector, window_width=1024, area_height=100)
            
                    final_display = np.vstack([display_img, text_area])
                    cv2.imshow('frame', final_display)
                else:
                    print("Error: Failed to process frame")
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print('FPS:', 1 / ((time.time() - start) / 10000))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', '-c', type=str, default='./cfg/models/X3D_8.yaml')
    parser.add_argument('--resume', '-r', type=str, default='/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/checkpoint0135.pth')
    args = parser.parse_args()
    main(args)