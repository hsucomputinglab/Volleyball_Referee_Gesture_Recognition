import cv2
import csv
import os

def annotate_serve_score_frames(video_path, output_csv='serve_score_annotations.csv'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    serve_frame = None
    annotations = []
    paused = False
    current_frame = 0
    frame = None

    print("==== 標註操作說明 ====")
    print("[s] 標註為『發球幀』")
    print("[d] 標註為『得分幀』（需先標註發球）")
    print("[Space] 快轉 100 幀")
    print("[p] 暫停/繼續播放")
    print("[q] 結束並儲存標註")
    print("======================")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("影片播放完畢")
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame is None:
            continue  # 等待第一幀

        display_frame = frame.copy()

        # 顯示目前幀數與標註狀態
        text = f"Frame: {current_frame}"
        if serve_frame is not None:
            text += f" | Serve: {serve_frame}"

        cv2.putText(display_frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Annotation Tool", display_frame)

        key = cv2.waitKey(0 if paused else 20) & 0xFF

        if key == ord('s'):
            serve_frame = current_frame
            print(f"✅ 發球幀標註於: {serve_frame}")
        elif key == ord('d'):
            if serve_frame is not None:
                score_frame = current_frame
                annotations.append([serve_frame, score_frame])
                print(f"✅ 得分幀標註於: {score_frame}（配對於 Serve: {serve_frame}）")
                serve_frame = None
            else:
                print("⚠️ 尚未標註發球幀，請先按 s")
        elif key == 32:  # 空白鍵快轉
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current + 100)
            paused = False
        elif key == ord('p'):
            paused = not paused
            print("⏸️ 暫停播放" if paused else "▶️ 繼續播放")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if annotations:
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['serve_frame', 'score_frame'])
            writer.writerows(annotations)
        print(f"✅ 標註結果已儲存至：{output_csv}")
    else:
        print("⚠️ 未進行任何標註，未建立 CSV 檔。")


if __name__ == "__main__":
    video_path = "/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Full Match ｜ Slovakia vs. Hungary ｜ CEV U18 Volleyball European Championship 2024 [3A9j3pNUanA].webm"
    annotate_serve_score_frames(video_path)


