import os
import shutil


json_folder = 'json-32'  # 存放 json 檔案的資料夾
video_folder = 'video-32'       # 目標資料夾

# 遍歷 json 資料夾中的所有檔案
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        # 取得檔案名（不包含副檔名）
        folder_name = filename.split('.')[0]
        
        # 設定找到對應目標子資料夾的標誌
        found = False
        
        # 遍歷 video 資料夾中的子資料夾
        for subfolder in os.listdir(video_folder):
            subfolder_path = os.path.join(video_folder, subfolder)
            
            # 確認這是資料夾
            if os.path.isdir(subfolder_path):
                target_subfolder = os.path.join(subfolder_path, folder_name)
                
                # 確認對應名稱的資料夾存在
                if os.path.exists(target_subfolder):
                    # 構造來源檔案路徑和目標檔案路徑
                    source_file = os.path.join(json_folder, filename)
                    target_file = os.path.join(target_subfolder, filename)
                    
                    # 移動檔案
                    shutil.move(source_file, target_file)
                    print(f'Moved {filename} to {target_subfolder}')
                    
                    # 設定找到對應資料夾的標誌
                    found = True
                    break
        
        # 如果沒有找到對應的目標資料夾，給出提示
        if not found:
            print(f'No target folder found for {filename}')
