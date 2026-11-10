# Same QIPs as 1_QIP_extraction.py, computed in parallel over chunks of 1,000 images; used for AVA (~255,000 images).
# Each chunk writes its own CSV (merged with 2_merge_QIP_chunks.py). Run from the Aesthetics Toolbox root folder.
import numpy as np
from PIL import Image
from skimage import color
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

from AT import balance_qips, CNN_qips, color_and_simple_qips, edge_entropy_qips,\
    fourier_qips, fractal_dimension_qips, PHOG_qips

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = 1e10

# Paths
results_path = "results/QIP/chunks/"
image_path = "data/AVA/images"
csv_base_name = "AVA_QIP"   # chunk number is appended
os.makedirs(results_path, exist_ok=True)

# Set all QIPs to True
check_dict = {key: True for key in [
    'Image size (pixels)', 'Aspect ratio', 'RMS contrast', 'Luminance entropy', 'Complexity',
    'Edge density', 'Color entropy', 'means RGB', 'means Lab', 'means HSV', 'std RGB', 'std Lab',
    'std HSV', 'Mirror symmetry', 'DCM', 'Balance', 'left-right', 'up-down',
    'left-right & up-down', 'Slope Redies', 'Slope Spehar', 'Slope Mather', 'Sigma',
    '2-dimensional', '3-dimensional', 'PHOG-based', 'CNN-based', 'Anisotropy',
    'Homogeneity', '1st-order', '2nd-order', 'Sparseness', 'Variability'
]}

# Load CNN kernel and bias (used by some AT functions)
[kernel, bias] = np.load(open("AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)

# Define header order (exactly as desired)
header_keys = [
    "img_file",
    "Image size (pixels)",
    "Aspect ratio",
    "RMS contrast",
    "Luminance entropy",
    "Complexity",
    "Edge density",
    "Color entropy",
    "mean R channel", "mean G channel", "mean B channel (RGB)",
    "mean L channel", "mean a channel", "mean b channel (Lab)",
    "mean H channel", "mean S channel", "mean V channel",
    "std R channel", "std G channel", "std B channel",
    "std L channel", "std a channel", "std b channel (Lab)",
    "std H channel", "std S channel", "std V channel",
    "Mirror symmetry",
    "DCM distance", "DCM x position", "DCM y position",
    "Balance",
    "CNN symmetry left-right", "CNN symmetry up-down", "CNN symmetry left-right & up-down",
    "Slope Redies", "Slope Spehar", "Slope Mather", "Fourier sigma",
    "2D Fractal dimension",
    "3D Fractal dimension",
    "Self-similarity (PHOG)",
    "Self-similarity (CNN)",
    "Anisotropy",
    "Homogeneity",
    "1st-order EOE", "2nd-order EOE",
    "Sparseness",
    "Variability"
]


def custom_round(num):
    """Round num: if below 1, round to 3 digits after the first nonzero digit; else round to 3 decimals."""
    if num < 1:
        sci = "{:e}".format(num)
        e_val = sci[-2:]
        return np.round(num, 3 + int(e_val))
    else:
        return np.round(num, 3)


def process_single_image(file):
    try:
        # Create result dictionary
        res_dict = {}
        res_dict["img_file"] = os.path.basename(file).replace(",", "_")

        # Load image representations
        pil_img = Image.open(file)
        img_rgb = np.asarray(pil_img.convert('RGB'))
        img_lab = color.rgb2lab(img_rgb)
        img_hsv = color.rgb2hsv(img_rgb)
        img_gray = np.asarray(pil_img.convert('L'))

        # Compute values:
        # 1. Image size (pixels)
        res_dict["Image size (pixels)"] = custom_round(color_and_simple_qips.image_size(img_rgb))
        # 2. Aspect ratio
        res_dict["Aspect ratio"] = custom_round(color_and_simple_qips.aspect_ratio(img_rgb))
        # 3. RMS contrast (using first channel std of Lab)
        res_dict["RMS contrast"] = custom_round(color_and_simple_qips.std_channels(img_lab)[0])
        # 4. Luminance entropy (L channel of Lab)
        res_dict["Luminance entropy"] = custom_round(color_and_simple_qips.shannonentropy_channels(img_lab[:, :, 0]))

        # 5-7. PHOG: Complexity, Self-similarity (PHOG), Anisotropy
        self_sim, complexity, anisotropy = PHOG_qips.PHOGfromImage(
            img_rgb, section=2, bins=16, angle=360, levels=3, re=-1, sesfweight=[1, 1, 1])
        res_dict["Complexity"] = custom_round(complexity)
        res_dict["Self-similarity (PHOG)"] = custom_round(self_sim)
        res_dict["Anisotropy"] = custom_round(anisotropy)

        # 8. Edge density, 45-46: 1st-order and 2nd-order EOE
        first_ord, sec_ord, edge_d = edge_entropy_qips.do_first_and_second_order_entropy_and_edge_density(img_gray)
        res_dict["Edge density"] = custom_round(edge_d)
        res_dict["1st-order EOE"] = custom_round(first_ord)
        res_dict["2nd-order EOE"] = custom_round(sec_ord)

        # 9. Color entropy (using hue channel of HSV)
        res_dict["Color entropy"] = custom_round(color_and_simple_qips.shannonentropy_channels(img_hsv[:, :, 0]))

        # 10-12. Means for RGB, Lab, HSV
        mean_rgb = color_and_simple_qips.mean_channels(img_rgb)
        res_dict["mean R channel"] = custom_round(mean_rgb[0])
        res_dict["mean G channel"] = custom_round(mean_rgb[1])
        res_dict["mean B channel (RGB)"] = custom_round(mean_rgb[2])

        mean_lab = color_and_simple_qips.mean_channels(img_lab)
        res_dict["mean L channel"] = custom_round(mean_lab[0])
        res_dict["mean a channel"] = custom_round(mean_lab[1])
        res_dict["mean b channel (Lab)"] = custom_round(mean_lab[2])

        # For HSV, use circular mean for H
        circ_mean, _ = color_and_simple_qips.circ_stats(img_hsv)
        mean_hsv = color_and_simple_qips.mean_channels(img_hsv)
        res_dict["mean H channel"] = custom_round(circ_mean)
        res_dict["mean S channel"] = custom_round(mean_hsv[1])
        res_dict["mean V channel"] = custom_round(mean_hsv[2])

        # 13-15. Standard deviations for RGB, Lab, HSV
        std_rgb = color_and_simple_qips.std_channels(img_rgb)
        res_dict["std R channel"] = custom_round(std_rgb[0])
        res_dict["std G channel"] = custom_round(std_rgb[1])
        res_dict["std B channel"] = custom_round(std_rgb[2])

        std_lab = color_and_simple_qips.std_channels(img_lab)
        res_dict["std L channel"] = custom_round(std_lab[0])
        res_dict["std a channel"] = custom_round(std_lab[1])
        res_dict["std b channel (Lab)"] = custom_round(std_lab[2])

        std_hsv = color_and_simple_qips.std_channels(img_hsv)
        # For H, use circular std:
        _, circ_std = color_and_simple_qips.circ_stats(img_hsv)
        res_dict["std H channel"] = custom_round(circ_std)
        res_dict["std S channel"] = custom_round(std_hsv[1])
        res_dict["std V channel"] = custom_round(std_hsv[2])

        # 16. Mirror symmetry
        res_dict["Mirror symmetry"] = custom_round(balance_qips.Mirror_symmetry(img_gray))

        # 17. DCM: returns three values
        dcm_vals = balance_qips.DCM(img_gray)
        res_dict["DCM distance"] = custom_round(dcm_vals[0])
        res_dict["DCM x position"] = custom_round(dcm_vals[1])
        res_dict["DCM y position"] = custom_round(dcm_vals[2])

        # 18. Balance
        res_dict["Balance"] = custom_round(balance_qips.Balance(img_gray))

        # 19. CNN symmetry: returns three values
        cnn_sym = CNN_qips.CNN_symmetry(img_rgb, kernel, bias)
        res_dict["CNN symmetry left-right"] = custom_round(cnn_sym[0])
        res_dict["CNN symmetry up-down"] = custom_round(cnn_sym[1])
        res_dict["CNN symmetry left-right & up-down"] = custom_round(cnn_sym[2])

        # 20. Fourrier measures: from fourier_qips.fourier_redies we get (sigma, slope)
        sigma, slope = fourier_qips.fourier_redies(img_gray, bin_size=2, cycles_min=10, cycles_max=256)
        # Desired order: Slope Redies, Slope Spehar, Slope Mather, Fourier sigma.
        res_dict["Slope Redies"] = custom_round(slope)
        # 21. Slope Spehar
        res_dict["Slope Spehar"] = custom_round(fourier_qips.fourier_slope_branka_Spehar_Isherwood(img_gray))
        # 22. Slope Mather
        res_dict["Slope Mather"] = custom_round(fourier_qips.fourier_slope_mather(img_rgb))
        # 23. Fourier sigma
        res_dict["Fourier sigma"] = custom_round(sigma)

        # 24. 2D Fractal dimension
        res_dict["2D Fractal dimension"] = custom_round(fractal_dimension_qips.fractal_dimension_2d(img_gray))
        # 25. 3D Fractal dimension
        res_dict["3D Fractal dimension"] = custom_round(fractal_dimension_qips.fractal_dimension_3d(img_gray))

        # 26. CNN-based self-similarity
        resp = CNN_qips.conv2d(img_rgb, kernel, bias)
        _, norm_map8 = CNN_qips.max_pooling(resp, patches=8)
        _, norm_map1 = CNN_qips.max_pooling(resp, patches=1)
        res_dict["Self-similarity (CNN)"] = custom_round(CNN_qips.CNN_selfsimilarity(norm_map1, norm_map8))

        # 27. Homogeneity (again, desired later in header)
        res_dict["Homogeneity"] = custom_round(balance_qips.Homogeneity(img_gray))

        # 28. Sparseness and 29. Variability using CNN_Variance
        resp = CNN_qips.conv2d(img_rgb, kernel, bias)
        _, norm_map_sp = CNN_qips.max_pooling(resp, patches=22)
        sparseness = CNN_qips.CNN_Variance(norm_map_sp, kind='sparseness')
        res_dict["Sparseness"] = custom_round(sparseness)

        _, norm_map_var = CNN_qips.max_pooling(resp, patches=12)
        variability = CNN_qips.CNN_Variance(norm_map_var, kind='variability')
        res_dict["Variability"] = custom_round(variability)

        return res_dict
    except Exception as e:
        print("Error processing", file, e)
        return None


# Get all image file paths
file_names = [os.path.join(dp, f) for dp, _, filenames in os.walk(image_path) for f in filenames]

# Optionally, if a base CSV exists, filter out already processed images
base_csv = os.path.join(results_path, csv_base_name + ".csv")
if os.path.exists(base_csv):
    df_exist = pd.read_csv(base_csv)
    exist_list = list(df_exist['img_file'])
    file_names = [fn for fn in file_names if os.path.basename(fn) not in exist_list]

if __name__ == '__main__':
    chunk_size = 1000
    current_chunk = []
    chunk_counter = 1
    total_processed = 0

    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_single_image, file_names),
                           total=len(file_names), desc="Processing Images"):
            if result is not None:
                current_chunk.append(result)
                total_processed += 1
            if len(current_chunk) >= chunk_size:
                csv_chunk_path = os.path.join(results_path, f"{csv_base_name}_chunk{chunk_counter}.csv")
                pd.DataFrame(current_chunk, columns=header_keys).to_csv(csv_chunk_path, index=False)
                print(f"Written {len(current_chunk)} records to {csv_chunk_path}")
                chunk_counter += 1
                current_chunk = []

    if current_chunk:
        csv_chunk_path = os.path.join(results_path, f"{csv_base_name}_chunk{chunk_counter}.csv")
        pd.DataFrame(current_chunk, columns=header_keys).to_csv(csv_chunk_path, index=False)
        print(f"Written {len(current_chunk)} records to {csv_chunk_path}")

    print(f"Processing completed. Total processed images: {total_processed}")
