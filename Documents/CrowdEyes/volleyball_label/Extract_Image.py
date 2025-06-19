import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, frame_indices_by_category, output_size=(1920, 1080)):
    """
    Extract specific frames from a video file, resize them to specified dimensions and save them by category.
    
    Args:
        video_path: Path to the video file
        output_folder: Base folder to save extracted frames
        frame_indices_by_category: Dictionary with category names as keys and lists of frame indices as values
        output_size: Tuple of (width, height) for the output images
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} total frames")
    print(f"Output size: {output_size[0]}x{output_size[1]}")
    
    # Create folders for each category
    for category in frame_indices_by_category.keys():
        category_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
    
    # Create a sorted list of all frame indices to extract
    all_frames = []
    for category, indices in frame_indices_by_category.items():
        for idx in indices:
            all_frames.append((idx, category))
    
    all_frames.sort(key=lambda x: x[0])
    
    # Extract frames
    current_frame = 0
    frames_extracted = 0
    
    for frame_idx, category in all_frames:
        # Skip to the frame (faster than reading each frame sequentially for sparse selections)
        if frame_idx < current_frame:
            # If we somehow went backwards, reset to the beginning
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0
        
        # Skip frames until we reach the target
        while current_frame < frame_idx:
            video.grab()  # Skip frame without decoding
            current_frame += 1
        
        # Read the target frame
        ret, frame = video.read()
        current_frame += 1
        
        if ret:
            # Resize the frame to the specified dimensions
            resized_frame = cv2.resize(frame, output_size)
            
            # Create output filename
            output_file = os.path.join(output_folder, category, f"{category}_frame_{frame_idx}.jpg")
            
            # Save the frame
            cv2.imwrite(output_file, resized_frame)
            frames_extracted += 1
            
            if frames_extracted % 10 == 0:
                print(f"Extracted {frames_extracted} frames so far...")
        else:
            print(f"Error: Could not read frame {frame_idx}")
    
    # Release the video
    video.release()
    print(f"Completed! Extracted {frames_extracted} frames in total.")


if __name__ == "__main__":
    # Set your video path here
    video_path = "/home/nckusoc/Documents/CrowdEyes/volleyball_label/C0032.MP4"
    
    # Set output folder
    output_folder = "extracted_frames"
    
    # Frame indices by category (as provided)
    frame_indices = {
        "left_hands_up" : [573,1676,3115,3473,6093,6361,15480,15856,16263,16562,17726,17960,22199,22517,23590,24013,25453,25792,26373,26639,28571,28895,31656,31980],

        "right_hands_up" : [1142,1418,3832,4042,4573,4804,5271,5592,6895,7190,7500,9340,9930,10245,10848,11342,11744,12071,13005,13386,13728,14563,15149,16837,17376,18489,18765,19359,19625,19994,21648,23012,23306,24402,25008,27002,27220,27709,27940,29290,29512,30006,30259,30889,31264,32611],

        "left_elbow"  : [617,3514,6389,15893,16585,17998,22559,24045,25852,26678,28927,32024],

        "right_elbow" : [1461,4085,4855,5654,7236,9383,10274,11378,12127,13074,13802,15216,17409,18839,19663,21703,23377,25046,27259,27997,29555,30308,31310 ],

        "stand" : [4410,5121,5919,6640,9647,10179,10483,10663,11603,12415,14007,17608,18382,19050,19260,21914,22017,22830,25313,26240,27545,28303,29172,29785,30631,31482 ],

        "ball_in" : [1717,5328,6154,6952,7543,9975,10903,11804,13423,16323,18529,19405,20033,22248,23048,24435,26432,27755,28637,29340,32645],

        "ball_out" : [1188,3168,3881,4622,15540,16878,17765,23647,27040,30918 ],

        "ball_touched" : [14619,30050 ]
        }
    
    # Extract frames with resizing to 1920x1080
    extract_frames(video_path, output_folder, frame_indices, output_size=(1920, 1080))