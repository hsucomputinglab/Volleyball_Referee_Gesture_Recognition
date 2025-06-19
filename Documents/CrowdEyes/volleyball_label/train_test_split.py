import os
import shutil
import random

# 設定來源資料夾和目標資料夾
video_folder = '/home/nckusoc/Documents/CrowdEyes/volleyball_label/video-32'       # 原始資料夾
output_folder = 'Videos-8c-32'     # 輸出資料夾，包含 train 和 test 資料夾

# 創建輸出資料夾及其子資料夾
train_folder = os.path.join(output_folder, '0')
test_folder = os.path.join(output_folder, '1')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 設定要分配的子資料夾範圍
subfolders_range = range(8)  # video/0 - video/7

# 收集所有子資料夾路徑
all_subfolders = []
for i in subfolders_range:
    subfolder_path = os.path.join(video_folder, str(i))
    if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
        all_subfolders.extend([os.path.join(subfolder_path, name) for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))])

# 打亂所有子資料夾
random.shuffle(all_subfolders)

# 定義訓練和測試資料夾的比例，這裡以8:2為例
train_ratio = 0.8
train_count = int(len(all_subfolders) * train_ratio)

# 分配子資料夾到 train 和 test 資料夾
for index, subfolder in enumerate(all_subfolders):
    if index < train_count:
        target_folder = train_folder
    else:
        target_folder = test_folder
    
    # 設定目標子資料夾路徑
    folder_name = os.path.basename(subfolder)
    target_subfolder = os.path.join(target_folder, folder_name)
    
    # 移動子資料夾到目標位置
    shutil.move(subfolder, target_subfolder)
    print(f'Moved {subfolder} to {target_subfolder}')
