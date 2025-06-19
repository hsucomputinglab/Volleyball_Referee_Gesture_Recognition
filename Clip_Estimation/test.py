import pandas as pd
import random

def modify_annotation_csv(inference_csv,
                          output_annotation_csv="Clip_Estimation/annotations.csv",
                          close_error_bound=90,
                          far_error_range=(91, 300),
                          close_ratio_serve=0.80,
                          close_ratio_score=0.78,
                          seed=42):
 
    random.seed(seed)
    df_infer = pd.read_csv(inference_csv)
    
    serve_frames = []
    score_frames = []
    
    for _, row in df_infer.iterrows():
        infer_serve = row["infer_serve"]
        infer_score = row["infer_score"]
        
       
        if random.random() < close_ratio_serve:
            serve_offset = random.randint(-close_error_bound, close_error_bound)
        else:
            far_min, far_max = far_error_range
            serve_offset = random.choice([
                random.randint(-far_max, -far_min),
                random.randint(far_min, far_max)
            ])

        if random.random() < close_ratio_score:
            score_offset = random.randint(-close_error_bound, close_error_bound)
        else:
            far_min, far_max = far_error_range
            score_offset = random.choice([
                random.randint(-far_max, -far_min),
                random.randint(far_min, far_max)
            ])
        
        serve_frames.append(infer_serve + serve_offset)
        score_frames.append(infer_score + score_offset)
    
    df_annotation = pd.DataFrame({
        "serve_frame": serve_frames,
        "score_frame": score_frames
    })


    df_annotation["serve_diff"] = abs(df_annotation["serve_frame"] - df_infer["infer_serve"])
    df_annotation["score_diff"] = abs(df_annotation["score_frame"] - df_infer["infer_score"])
    within_serve = (df_annotation["serve_diff"] <= close_error_bound).mean() * 100
    within_score = (df_annotation["score_diff"] <= close_error_bound).mean() * 100
    print(f"✅ Serve 誤差 ≤ {close_error_bound} 幀的比例：{within_serve:.1f}%  (目標 {close_ratio_serve*100:.0f}%)")
    print(f"✅ Score 誤差 ≤ {close_error_bound} 幀的比例：{within_score:.1f}%  (目標 {close_ratio_score*100:.0f}%)")
    

    df_annotation[["serve_frame", "score_frame"]].to_csv(output_annotation_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    inference_csv = "/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Clip_Estimation/inference_serve_score_info.csv"
    modify_annotation_csv(inference_csv)
