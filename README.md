# 🏷️ Volleyball Referee Labeling Toolkit

This submodule provides tools for annotating, formatting, and preprocessing referee gesture data in volleyball match videos. It supports label generation in `.txt` and `.json` formats, dataset splitting, and conversion for model training.

---

## 📁 File Overview
```
| File / Script              | Description                                       |
|---------------------------|---------------------------------------------------|
| `Extract_Image.py`        | Extracts frames from video                        |
| `frames2folder.py`        | Organizes extracted frames into folders           |
| `find-referee.py`         | Detects referee region in frames                  |
| `image.py`                | Image-based preprocessing (resizing, cropping)    |
| `file_txt.py`             | Generates `.txt` labels from annotations          |
| `json2file.py`            | Converts JSON labels to model-readable `.txt`     |
| `create_json.py`          | Creates `.json` annotation files                  |
| `merge-data.py`           | Merges labels from multiple sessions              |
| `train_test_split.py`     | Splits data into train/test sets                  |
| `new-inference.py`        | Inference on new data using labeling model        |
| `inference.py`            | Inference pipeline entry point                    |
| `Volley.txt`              | Label definitions or class mapping                |
| `output-15.txt`           | Sample label file (TVL Video 15)                  |
| `output-26.txt`           | Sample label file (TVL Video 26)                  |
| `output-32.txt`           | Sample label file (TVL Video 32)                  |
| `裁判手勢label-15.txt`     | Label content in Chinese                          |
| `裁判手勢label-26.txt`     | Label content in Chinese                          |
| `裁判手勢label-32.txt`     | Label content in Chinese                          |
```

---

## 🧪 Labeling Workflow
```
- python image.py
影片拆成一張張影像

-python find-referee.py
先滑鼠標註裁判

- python inference.py
模型偵測裁判bounding box位置

- python create-json.py
根據標註裁判類別，產生對應json檔案

- python Extract_Image.py
根據標註裁判類別幀數，前後取20張影像

- python json2file.py
將json檔案一一對應到影像sequence folder中

- python merge-data.py
將各影片處理後的資料集整合一起

- python train_test_split.py
分成訓練和測試集

