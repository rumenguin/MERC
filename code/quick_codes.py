import pandas as pd

# Define paths
base_input_path = "/Users/rumenguin/Research/MERC/EmoReact/dataset.csv"
base_output_path = "/Users/rumenguin/Research/MERC/EmoReact/emotions.csv"

# Load the dataset
df = pd.read_csv(base_input_path)

# Replace missing values in 'Labels' with the string 'None'
df['Labels'] = df['Labels'].fillna('None')

# Extract required columns
df_emotions = df[['Video', 'Labels', 'Valence']].copy()

# Save to new emotions.csv
df_emotions.to_csv(base_output_path, index=False)

print("✅ emotions.csv created with string 'None' for missing Labels and no missing values.")
