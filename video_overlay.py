import cv2
import pandas as pd
import numpy as np
import os

def add_behavior_overlay(video_path, csv_data, start_time=8.04, frame_interval=2.68):
    # Check if files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(csv_data):
        raise FileNotFoundError(f"CSV file not found: {csv_data}")
        
    # Read the CSV data
    try:
        df = pd.read_csv(csv_data)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}. Please check the file format and path.")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        raise Exception("Could not determine video FPS. Please check the video file.")
    
    # Create output filename
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    output_path = os.path.join(video_dir, f'output_{video_name}')
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise Exception("Failed to create output video file. Please check write permissions and codec support.")
    
    frame_count = 0
    csv_index = 0
    
    # Initialise current behavior and confidence
    current_behavior = "UNKNOWN"
    current_confidence = 100.0  # Changed to 100.0
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved as: {output_path}")
    print(f"Start time: {start_time} seconds")
    print(f"Frame interval: {frame_interval} seconds")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current time in video
            current_time = frame_count / fps
            
            # Check if we should display any overlay yet
            if current_time >= start_time:
                # Calculate which CSV row we should be showing based on time
                time_since_start = current_time - start_time
                csv_index = int(time_since_start / frame_interval)
                
                # Update behavior and confidence if we have data for this time
                if csv_index < len(df):
                    current_behavior = df.iloc[csv_index]['Classified Behavior'].upper()
                    current_confidence = df.iloc[csv_index]['Confidence']
                
            # Always display the current behavior (now showing from start of video)
            text = f"{current_behavior}: {current_confidence:.1f}%"
            
            # Calculate position for bottom left (with some padding)
            text_x = 20
            text_y = height - 40
            
            # Add black background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
            )
            cv2.rectangle(
                frame,
                (text_x - 10, text_y + 10),
                (text_x + text_width + 10, text_y - text_height - 10),
                (0, 0, 0),
                -1
            )
            
            # Add text
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,  # Font scale
                (255, 255, 255),  # White color
                2,  # Thickness
                cv2.LINE_AA
            )
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processing frame {frame_count} (Time: {current_time:.2f}s)")
            
    except Exception as e:
        raise Exception(f"Error during video processing: {str(e)}")
        
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    print("Processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total video duration: {frame_count/fps:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Add behavior overlay to video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('csv_path', help='Path to the CSV file')
    parser.add_argument('--start-time', type=float, default=8.04, help='Time to start showing behaviors (default: 8.04)')
    parser.add_argument('--frame-interval', type=float, default=2.68, help='Seconds between behavior updates (default: 2.68)')
    
    args = parser.parse_args()
    
    try:
        add_behavior_overlay(args.video_path, args.csv_path, args.start_time, args.frame_interval)
    except Exception as e:
        print(f"Error: {str(e)}")
