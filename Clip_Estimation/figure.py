import pandas as pd
import matplotlib.pyplot as plt
import os


input_csv = "/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Clip_Estimation/frame_comparison_result_all.csv"  
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# 讀取資料
df = pd.read_csv(input_csv)

# 加入 ±90 幀的合理範圍欄位
df['serve_within_150'] = df['serve_diff'] <= 90
df['score_within_150'] = df['score_diff'] <= 90

# 圖1：散佈圖
plt.figure(figsize=(12, 6))
plt.scatter(range(len(df)), df['serve_diff'], color='blue', label='Serve Difference')
plt.scatter(range(len(df)), df['score_diff'], color='green', label='Score Difference')
plt.axhline(y=90, color='red', linestyle='--', label='±90 Frame Threshold')
plt.title('Frame Difference Comparison per Video (±90)')
plt.xlabel('Video Index')
plt.ylabel('Frame Difference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frame_diff_scatter_90.png"))
plt.close()

# 圖2：發球誤差圓餅圖
labels_serve = ['Within 90', 'Outside 90']
sizes_serve = [df['serve_within_90'].sum(), len(df) - df['serve_within_90'].sum()]

plt.figure(figsize=(6, 6))
plt.pie(sizes_serve, labels=labels_serve, autopct='%1.1f%%', startangle=90)
plt.title('Serve Difference Distribution (±90)')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "serve_diff_pie_90.png"))
plt.close()

# 圖3：得分誤差圓餅圖
labels_score = ['Within 90', 'Outside 90']
sizes_score = [df['score_within_90'].sum(), len(df) - df['score_within_90'].sum()]

plt.figure(figsize=(6, 6))
plt.pie(sizes_score, labels=labels_score, autopct='%1.1f%%', startangle=90)
plt.title('Score Difference Distribution (±90)')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "score_diff_pie_90.png"))
plt.close()
