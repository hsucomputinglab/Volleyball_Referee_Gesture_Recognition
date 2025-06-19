import cv2
import os

# Open the video file
cap = cv2.VideoCapture('/home/nckusoc/Documents/CrowdEyes/volleyball_label/C0032.MP4')

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()


# Directory to save frames
output_directory = '0032-frames'
os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

max_frames = 33120

frame_count = 0
while True:
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Read a frame from the video
    ret, frame = cap.read()

    # if idx % 2 == 0:
    #     continue
    
    # Check if the frame was read successfully
    if not ret:
        break

    if idx >= max_frames:
        break
    
    # Save the frame as an image
    filename = os.path.join(output_directory, f"{frame_count}.jpg") 
    frame = cv2.resize(frame, (1920, 1080))
    # cv2.imshow('frame', frame)
    cv2.imwrite(filename, frame)
    
    # Print status
    print(f"Frame {frame_count} saved as {filename}")
    
    # Increment frame count
    frame_count += 1
    if(frame_count % 10000 == 0):
         print(f"Progress: {frame_count/max_frames*100:.1f}%")

# Release the video capture object and close the video file
cap.release()

