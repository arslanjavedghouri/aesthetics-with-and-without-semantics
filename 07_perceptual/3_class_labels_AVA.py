import pandas as pd

# Adds the 5-class label (bins 1-2 -> 1, ..., 9-10 -> 5, from the truncated-Gaussian mean MU) to the QIP table of all AVA images.
csv_df = pd.read_csv(r'results/QIP/AVA_QIP.csv')   # from 2_merge_QIP_chunks.py

# Extract integer image names from 'img_file'
csv_df['Image ID'] = csv_df['img_file'].str.replace('.jpg', '', regex=False).astype(int)

xlsx_df = pd.read_excel(r"results/AVA_ratings_with_MU.xlsx")   # from 01_dataset_preparation/1_truncated_gaussian_mu_AVA.py

# Merge CSV and XLSX data on 'Image ID'
merged_df = pd.merge(csv_df, xlsx_df[['Image ID', 'MU']], on='Image ID', how='left')

# assign the class from the truncated-Gaussian mean (MU)
def get_class(mu_value):
    if 0 <= mu_value < 2:
        return 1
    elif 2 <= mu_value < 4:
        return 2
    elif 4 <= mu_value < 6:
        return 3
    elif 6 <= mu_value < 8:
        return 4
    elif 8 <= mu_value <= 9.99:
        return 5
    else:
        return None

# Apply class assignment
merged_df['Class Name'] = merged_df['MU'].apply(get_class)

# Insert 'Class Name' column right after 'img_file'
img_file_index = merged_df.columns.get_loc('img_file')
cols = list(merged_df.columns)
# Move Class Name to be right after img_file
cols.insert(img_file_index + 1, cols.pop(cols.index('Class Name')))
final_df = merged_df[cols]

# drop the columns that are not needed
# image size and aspect ratio are not among the 45 QIPs used (Supplementary Table S3)
final_df = final_df.drop(columns=['MU', 'Image ID', 'Image size (pixels)', 'Aspect ratio'], errors='ignore')

# Save to new CSV
final_df.to_csv(r'results/QIP/AVA_QIP_with_class.csv', index=False)
