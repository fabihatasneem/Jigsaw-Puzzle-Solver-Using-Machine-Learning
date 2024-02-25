import numpy as np
import pandas as pd

# Original CSV file path
original_csv_path = "F:\\archive\\data\\puzzle_2x2\\train.csv"

# Load the original CSV file
df = pd.read_csv(original_csv_path)

# Define the labels (permutation of [0, 1, 2, 3])
labels = ['0 1 2 3', '0 1 3 2', '0 2 1 3', '0 2 3 1', '0 3 1 2', '0 3 2 1',
          '1 0 2 3', '1 0 3 2', '1 2 0 3', '1 2 3 0', '1 3 0 2', '1 3 2 0',
          '2 0 1 3', '2 0 3 1', '2 1 0 3', '2 1 3 0', '2 3 0 1', '2 3 1 0',
          '3 0 1 2', '3 0 2 1', '3 1 0 2', '3 1 2 0', '3 2 0 1', '3 2 1 0']

# Create an empty DataFrame to store filtered data
filtered_df = pd.DataFrame(columns=['image', 'label'])

# Define the desired number of images per label class
num_images_per_class = 900

# Dictionary to store the count of images per label class
label_counts = {label: 0 for label in labels}

# Iterate over each label class and filter the data
for label in labels:
    # Split the label string into individual numbers
    label_values = list(map(int, label.split()))

    # Filter rows containing the current label
    filtered_rows = df[df['label'].apply(lambda x: list(map(int, x.split())) == label_values)]

    # Take the first 'num_images_per_class' rows
    filtered_rows = filtered_rows.head(num_images_per_class)

    # Update the count of images for the current label class
    label_counts[label] = len(filtered_rows)

    # Append filtered rows to the new DataFrame
    filtered_df = pd.concat([filtered_df, filtered_rows], ignore_index=True)

# Save the filtered DataFrame to a new CSV file
filtered_csv_path = "F:\\archive\\data\\puzzle_2x2\\train_filtered.csv"
filtered_df.to_csv(filtered_csv_path, index=False)

print("Filtered CSV file saved successfully.")

# Print the count of images in each label class
for label, count in label_counts.items():
    print(f"Label {label}: {count} images")
