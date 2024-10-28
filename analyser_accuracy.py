import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class BehaviorAccuracyAnalyzer:
    def __init__(self, predicted_file, true_behavior_file):
        """
        Initialize the analyzer with predicted and true behavior files
        
        Args:
            predicted_file (str): Path to the CSV file with predicted behaviors
            true_behavior_file (str): Path to the CSV file with true behaviors
        """
        self.predicted_data = pd.read_csv(predicted_file)
        self.true_data = pd.read_csv(true_behavior_file)
        self.behaviors = ['sitting', 'walking', 'climbing']
        
    def prepare_data(self):
        """Prepare and align the predicted and true behavior data"""
        # Ensure we're comparing the same frames
        merged_data = pd.merge(
            self.predicted_data,
            self.true_data,
            left_on='Frame',
            right_on='Frames',
            how='inner'
        )
        
        self.y_true = merged_data['True_Behaviour'].str.lower()
        self.y_pred = merged_data['Classified Behavior'].str.lower()
        
        return merged_data
        
    def calculate_metrics(self):
        """Calculate various accuracy metrics"""
        # Overall accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Detailed classification report
        report = classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True
        )
        
        # Convert classification report to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Confusion matrix
        conf_matrix = confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=self.behaviors
        )
        
        return accuracy, report_df, conf_matrix
    
    def plot_confusion_matrix(self, conf_matrix):
        """Plot confusion matrix as a heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.behaviors,
            yticklabels=self.behaviors
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Behavior')
        plt.xlabel('Predicted Behavior')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    def analyze_frame_by_frame(self, merged_data):
        """Analyze and print frame-by-frame comparison"""
        mismatches = merged_data[merged_data['Classified Behavior'].str.lower() != 
                               merged_data['True_Behaviour'].str.lower()]
        
        return mismatches
    
    def run_analysis(self):
        """Run the complete analysis and save results"""
        # Prepare data
        merged_data = self.prepare_data()
        
        # Calculate metrics
        accuracy, report_df, conf_matrix = self.calculate_metrics()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(conf_matrix)
        
        # Get mismatched frames
        mismatches = self.analyze_frame_by_frame(merged_data)
        
        # Save results to files
        report_df.to_csv('classification_report.csv')
        mismatches.to_csv('mismatched_frames.csv', index=False)
        
        # Print summary results
        print("\n=== Behavior Classification Analysis ===")
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(report_df)
        
        print("\nMismatched Frames:")
        if len(mismatches) > 0:
            print(f"Found {len(mismatches)} mismatched frames:")
            for _, row in mismatches.iterrows():
                print(f"Frame {row['Frame']}: Predicted '{row['Classified Behavior']}' "
                      f"but was actually '{row['True_Behaviour']}'")
        else:
            print("No mismatched frames found!")
        
        print("\nResults have been saved to:")
        print("- classification_report.csv")
        print("- mismatched_frames.csv")
        print("- confusion_matrix.png")
        
        return accuracy, report_df, conf_matrix, mismatches

if __name__ == "__main__":
    analyzer = BehaviorAccuracyAnalyzer('BehaviorAnalysis.csv', 'true_behaviour.csv')
    analyzer.run_analysis()