import os
import shutil
from pathlib import Path

def merge_video_classes():
    # 來源資料夾
    src_folders = ['Videos-8c-15', 'Videos-8c-26', 'Videos-8c-32']
    # 要處理的子資料夾
    subfolders = ['0', '1']
    
    # 建立輸出目錄
    output_base = "merged_8class_videos"
    os.makedirs(output_base, exist_ok=True)
    
    # 處理每個子資料夾
    for subfolder in subfolders:
        output_dir = os.path.join(output_base, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集並合併子資料夾內容
        for src_folder in src_folders:
            src_path = os.path.join(src_folder, subfolder)
            if not os.path.exists(src_path):
                print(f"警告：資料夾不存在 {src_path}")
                continue
                
            try:
                # 列出資料夾內容
                items = os.listdir(src_path)
                for item in items:
                    src_item_path = os.path.join(src_path, item)
                    dest_item_path = os.path.join(output_dir, item)
                    
                    # 如果是資料夾，複製整個資料夾
                    if os.path.isdir(src_item_path):
                        if os.path.exists(dest_item_path):
                            if src_folder == 'Videos-8c-15':
                                shutil.rmtree(dest_item_path)
                                shutil.copytree(src_item_path, dest_item_path)
                                print(f"更新資料夾: {item} (從 {src_folder})")
                        else:
                            shutil.copytree(src_item_path, dest_item_path)
                            print(f"新增資料夾: {item} (從 {src_folder})")
                    
                    # 如果是檔案，直接複製
                    elif os.path.isfile(src_item_path):
                        if os.path.exists(dest_item_path):
                            if src_folder == 'Videos-8c-15':
                                shutil.copy2(src_item_path, dest_item_path)
                                print(f"更新檔案: {item} (從 {src_folder})")
                        else:
                            shutil.copy2(src_item_path, dest_item_path)
                            print(f"新增檔案: {item} (從 {src_folder})")
                            
            except Exception as e:
                print(f"處理資料夾 {src_path} 時發生錯誤: {str(e)}")
                continue
        
        print(f"資料夾 {subfolder} 合併完成")

if __name__ == "__main__":
    try:
        merge_video_classes()
        print("所有資料夾合併完成！")
    except Exception as e:
        print(f"程式執行時發生錯誤: {str(e)}")