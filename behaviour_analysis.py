import pandas as pd
import numpy as np
from math import atan2, degrees, sqrt

class OrangutanBehaviorAnalyzer:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.behaviors = ['sitting', 'walking', 'climbing']  # Removed 'swinging', renamed 'climbing'
        
    def calculate_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        angle = degrees(atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2))
        return abs(angle)
    
    def get_joint_coordinates(self, row, joint):
        """Get x,y coordinates for a specific joint"""
        return row[f'{joint}_x'], row[f'{joint}_y']
    
    def calculate_velocity(self, current_frame, previous_frame, joint):
        """Calculate change in position between frames"""
        if previous_frame is None:
            return 0
        
        curr_x, curr_y = self.get_joint_coordinates(current_frame, joint)
        prev_x, prev_y = self.get_joint_coordinates(previous_frame, joint)
        
        return self.calculate_distance(curr_x, curr_y, prev_x, prev_y)
    
    def calculate_centre_hip(self, left_hip_x, left_hip_y, right_hip_x, right_hip_y):
        """Calculate the center point between the left and right hip coordinates."""
        center_hip_x = (left_hip_x + right_hip_x) / 2
        center_hip_y = (left_hip_y + right_hip_y) / 2

        return center_hip_x, center_hip_y
    
    def analyze_sitting(self, row, prev_row=None):
        """Analyze sitting behavior confidence"""
        confidence = 0
        
        # Check if hips are relatively close to knees (sitting position)
        left_hip_to_knee = self.calculate_distance(
            row['left_hip_x'], row['left_hip_y'],
            row['left_knee_x'], row['left_knee_y']
        )
        right_hip_to_knee = self.calculate_distance(
            row['right_hip_x'], row['right_hip_y'],
            row['right_knee_x'], row['right_knee_y']
        )
        
        # Calculate the center of the hips
        center_hip_x, center_hip_y = self.calculate_centre_hip(
            row['left_hip_x'], row['left_hip_y'],
            row['right_hip_x'], row['right_hip_y']
        )

        # Check torso verticality
        torso_angle = self.calculate_angle(
            (row['head_x'], row['head_y']),
            (row['torso_x'], row['torso_y']),
            (center_hip_x, center_hip_y)
        )
        
        # Low movement between frames indicates sitting
        if prev_row is not None:
            movement = self.calculate_velocity(row, prev_row, 'torso')
            if movement < 50:  # threshold for low movement
                confidence += 35
        
        # Short hip-to-knee distance indicates bent legs (sitting)
        if left_hip_to_knee < 100 and right_hip_to_knee < 100:
            confidence += 35
            
        # Vertical torso indicates sitting
        if 100 < torso_angle < 178:
            confidence += 30
            
        return min(confidence, 100)
    
    def analyze_climbing(self, row, prev_row=None):
        """Analyze climbing behavior confidence (previously swinging)"""
        confidence = 0
        
        # Check arm extension and position
        left_arm_extension = self.calculate_distance(
            row['left_shoulder_x'], row['left_shoulder_y'],
            row['left_hand_x'], row['left_hand_y']
        )
        right_arm_extension = self.calculate_distance(
            row['right_shoulder_x'], row['right_shoulder_y'],
            row['right_hand_x'], row['right_hand_y']
        )
        
        # Check if both hands are elevated (key indicator for climbing)
        hands_elevated = (
            row['left_hand_y'] < row['torso_y'] and 
            row['right_hand_y'] < row['torso_y']
        )
        
        # Check feet elevation (should be free-hanging for climbing)
        feet_hanging = (
            row['left_foot_y'] > row['torso_y'] and 
            row['right_foot_y'] > row['torso_y']
        )
        
        # High arm extension indicates climbing
        if left_arm_extension > 125 or right_arm_extension > 125:
            confidence += 40
            
        # Elevated hands is a strong indicator of climbing
        if hands_elevated:
            confidence += 30
            
        # Free-hanging feet
        if feet_hanging:
            confidence += 30
            
        return min(confidence, 100)
    
    def analyze_walking(self, row, prev_row=None):
        """Analyze walking behavior confidence"""
        confidence = 0
        
        # Check leg separation
        leg_separation = self.calculate_distance(
            row['left_foot_x'], row['left_foot_y'],
            row['right_foot_x'], row['right_foot_y']
        )
        
        # Check horizontal movement between frames
        if prev_row is not None:
            horizontal_movement = abs(row['torso_x'] - prev_row['torso_x'])
            if 90 < horizontal_movement < 1000:
                confidence += 50
        
        # Moderate leg separation indicates walking
        if 115 < leg_separation < 125:
            confidence += 50
            
        return min(confidence, 100)
    
    def analyze_frame(self, frame_idx):
        """Analyze a single frame and return behavior confidences"""
        current_row = self.data.iloc[frame_idx]
        previous_row = self.data.iloc[frame_idx - 1] if frame_idx > 0 else None
        
        confidences = {
            'sitting': self.analyze_sitting(current_row, previous_row),
            'walking': self.analyze_walking(current_row, previous_row),
            'climbing': self.analyze_climbing(current_row, previous_row)  # Renamed from swinging
        }
        
        # Normalize confidences to sum to 100%
        total = sum(confidences.values())
        if total > 0:
            confidences = {k: (v / total) * 100 for k, v in confidences.items()}
            
        return confidences

    def analyze_all_frames(self, output_csv='BehaviorAnalysis.csv'):
        """Analyze all frames, print results, and save to a CSV file"""
        results = []  # List to store analysis results for saving to CSV

        for frame_idx in range(len(self.data)):
            confidences = self.analyze_frame(frame_idx)
            
            # Determine majority behavior and its confidence
            majority_behavior, majority_confidence = max(confidences.items(), key=lambda x: x[1])
            
            # If the majority behavior confidence is less than 40%, set it to "unknown"
            if majority_confidence < 40:
                majority_behavior = 'unknown'
            
            # Print results to terminal
            print(f"\nFrame {frame_idx + 1}:")
            for behavior, confidence in confidences.items():
                print(f"{behavior.capitalize()}: {confidence:.1f}%")
            print(f"Classified Behavior: {majority_behavior.upper()}")
            
            # Save the results for this frame into a dictionary
            frame_result = {
                'Frame': frame_idx + 1,
                'Sitting': confidences['sitting'],
                'Walking': confidences['walking'],
                'Climbing': confidences['climbing'],
                'Classified Behavior': majority_behavior,
                'Confidence': majority_confidence
            }
            results.append(frame_result)

        # Convert the results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

# Usage example
analyzer = OrangutanBehaviorAnalyzer('ProcessedData.csv')
analyzer.analyze_all_frames()
