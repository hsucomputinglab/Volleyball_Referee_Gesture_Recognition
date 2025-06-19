# Volleyball Referee Gesture Recognition 🏐✋
This project focuses on recognizing volleyball referee gestures from video . The system supports real-time inference, gesture recognition, and automatic match segmentation.


## 📁 Project Structure
```
Volleyball_Referee_Gesture_Recognition/
├── cfg/ # Configuration files
├── src/ # Source code for model and pipeline
├── export/ # Scripts for exporting results
├── Clip_Estimation/ # Referee gesture estimation module
├── FPS_TEST/ # FPS testing and performance analysis
├── output_images/ # Inference result visualizations
├── main.py # Main execution script
├── requirements.txt # Required Python packages
```

- ✅ Real-time referee gesture recognition
- 🎬 Match segmentation based on serve and score sequence gestures
- 📊 Score tracking and game state updates
- 🧠 Custom training pipeline for 6–8 gesture classes： Left hand up， right hand up, Left elbow, Right elbow, stand, ball in, ball out, ball touched.

## 🛠️ Installation
```bash
# Clone the repository
git clone git@github.com:hsucomputinglab/Volleyball_Referee_Gesture_Recognition.git
cd Volleyball_Referee_Gesture_Recognition

# Install dependencies
pip install -r requirements.txt
```

##  📦 Dataset: Datasets are not included due to size limitations.
```
- Download the dataset here: (目前還在整理檔案太大）
- dataset structure
 - Youtube-5c:   hand up, elbow, stand, point, no meaning
 - Youtube-6c:   Left hand up, Right hand up, Left elbow, Right elbow, stand, point
 - TVL-8c:       Left hand up, Right hand up, Left elbow, Right elbow, stand, ball in, ball out, ball touched
```

## 📓 Usage：
```
model Training： torchrun main.py -c cfg/models/X3D_4816.yaml 
- 可修改設定檔：cfg/models/X3D_4816.yaml
model Inference： python export/Clip.py 
-  可修改推論腳本：export/Clip.py
```





