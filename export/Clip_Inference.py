import pandas as pd

def compare_csv_inference_with_annotations(infer_csv, annotation_csv, output_csv='Clip_Estimation/frame_comparison_result_all.csv'):
 
    infer_df = pd.read_csv(infer_csv)
    gt_df = pd.read_csv(annotation_csv)
    gt_df = gt_df.sort_values(by='serve_frame').reset_index(drop=True)

    n = min(len(infer_df), len(gt_df))

    results = []
    for i in range(n):
        video_file = infer_df.loc[i, 'video_file']
        infer_serve = infer_df.loc[i, 'infer_serve']
        infer_score = infer_df.loc[i, 'infer_score']
        gt_serve = gt_df.loc[i, 'serve_frame']
        gt_score = gt_df.loc[i, 'score_frame']

        serve_diff = abs(infer_serve - gt_serve)
        score_diff = abs(infer_score - gt_score)

        serve_within_90 = serve_diff <= 90
        score_within_90 = score_diff <= 90

        results.append({
            'video_file': video_file,
            'gt_serve': gt_serve,
            'infer_serve': infer_serve,
            'serve_diff': serve_diff,
            'serve_within_90': serve_within_90,
            'gt_score': gt_score,
            'infer_score': infer_score,
            'score_diff': score_diff,
            'score_within_90': score_within_90
        })

    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 對比結果已儲存至：{output_csv}")


if __name__ == '__main__':
    infer_csv = '/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Clip_Estimation/inference_serve_score_info.csv'
    annotation_csv = '/home/nckusoc/Documents/CrowdEyes/Volleyball_Referee/Clip_Estimation/annotations.csv'
    compare_csv_inference_with_annotations(infer_csv, annotation_csv)
