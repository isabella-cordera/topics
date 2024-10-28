import pandas as pd

def preprocess_csv(input_csv, output_csv):
    # Load the CSV file
    data = pd.read_csv(input_csv)
    
    # Debug: Print first few rows to check if data is loaded correctly
    print("Initial CSV Data:")
    print(data.head())
    
    # Step 1: Drop the first two rows (metadata) and reset the index
    data_cleaned = data.drop([0, 1]).reset_index(drop=True)
    
    # Step 2: Drop unnecessary metadata columns (like 'scorer', 'Unnamed: 1', etc.)
    # Keep only columns that start with 'test' which hold the actual data
    data_cleaned = data_cleaned.loc[:, data_cleaned.columns.str.startswith('test')]
    
    # Step 3: Define the correct column names based on body parts
    column_names = [
        "head_x", "head_y", "chest_x", "chest_y", "torso_x", "torso_y",
        "left_shoulder_x", "left_shoulder_y", "left_elbow_x", "left_elbow_y", 
        "left_hand_x", "left_hand_y", "right_shoulder_x", "right_shoulder_y", 
        "right_elbow_x", "right_elbow_y", "right_hand_x", "right_hand_y", 
        "left_hip_x", "left_hip_y", "left_knee_x", "left_knee_y", "left_foot_x", 
        "left_foot_y", "right_hip_x", "right_hip_y", "right_knee_x", "right_knee_y", 
        "right_foot_x", "right_foot_y"
    ]
    
    # Step 4: If there are extra columns, remove them
    if len(data_cleaned.columns) > len(column_names):
        print(f"Warning: Found {data_cleaned.shape[1]} columns, trimming to {len(column_names)} columns.")
        data_cleaned = data_cleaned.iloc[:, :len(column_names)]
    
    # Step 5: Assign the correct column names
    data_cleaned.columns = column_names
    
    # Debug: Print to check if `head_x` has valid data before conversion
    print("Data for head_x column (before conversion):")
    print(data_cleaned['head_x'].head())

    # Step 6: Convert all columns to numeric values
    for col in data_cleaned.columns:
        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
    
    # Debug: Print to check if `head_x` has valid data after conversion
    print("Data for head_x column (after conversion):")
    print(data_cleaned['head_x'].head())
    
    # Step 7: Save the cleaned data to a new CSV file
    data_cleaned.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to {output_csv}")

# Example usage:
if __name__ == "__main__":
    input_csv = 'CollectedData_test.csv'  # Path to your initial CSV file
    output_csv = 'ProcessedData.csv'  # Path where the cleaned CSV file will be saved
    preprocess_csv(input_csv, output_csv)
