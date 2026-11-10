import pandas as pd
import os

# Adds the MSC class (bin 1-5 = image folder) to the QIP table.
main_dir = r"data/MSC/images"          # folders 1-5

csv_path = r'results/QIP/MSC_QIP.csv'   # from 1_QIP_extraction.py
df = pd.read_csv(csv_path)

# Subfolder names to search in
folders = ['1', '2', '3', '4', '5']

# Function to find the folder (class name) that contains the image
def find_class_name(img_name):
    for folder in folders:
        image_path = os.path.join(main_dir, folder, img_name)
        if os.path.isfile(image_path):
            return folder
    return None  # If image is not found in any folder

# Generate class names
class_names = df['img_file'].apply(find_class_name)

# Insert 'Class Name' column right after 'img_file'
img_file_index = df.columns.get_loc('img_file')
df.insert(img_file_index + 1, 'Class Name', class_names)

# image size and aspect ratio are not among the 45 QIPs used (Supplementary Table S3)
df = df.drop(columns=['Image size (pixels)', 'Aspect ratio'], errors='ignore')

# Save the updated CSV
output_path = r'results/QIP/MSC_QIP_with_class.csv'
df.to_csv(output_path, index=False)
