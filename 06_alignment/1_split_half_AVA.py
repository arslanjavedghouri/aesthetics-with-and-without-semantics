import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import ast
from tqdm import tqdm
import os
import sys
from datetime import datetime
import random
import zlib


class Logger:
    """Class to log output to both terminal and file"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def parse_histogram(hist_str):
    """Parse histogram string to numpy array"""
    if isinstance(hist_str, str):
        return np.array(ast.literal_eval(hist_str))
    elif isinstance(hist_str, list):
        return np.array(hist_str)
    else:
        return hist_str


def normalize_histogram(hist):
    """Normalize histogram to probability distribution"""
    hist = hist.astype(float)
    total = hist.sum()

    if total == 0:
        return hist
    return hist / total


def compute_emd(hist1, hist2):
    """Compute Earth Mover's Distance (Wasserstein distance) between two histograms"""
    p1 = normalize_histogram(hist1)
    p2 = normalize_histogram(hist2)

    bins = np.arange(len(p1))
    emd = wasserstein_distance(bins, bins, p1, p2)
    return emd


def compute_bhattacharyya(hist1, hist2):
    """Compute Bhattacharyya coefficient between two histograms"""
    p1 = normalize_histogram(hist1)
    p2 = normalize_histogram(hist2)

    bc = np.sum(np.sqrt(p1 * p2))
    return bc


def normalize_emd_to_similarity(emd, max_possible_emd):
    """
    Convert EMD to normalized similarity score (0-1).
    0 = bad overlap (maximum distance)
    1 = identical overlap (zero distance)

    Parameters:
    - emd: Earth Mover's Distance value
    - max_possible_emd: Maximum possible EMD for normalization (typically number of bins - 1)

    Returns:
    - Normalized similarity score between 0 and 1
    """
    if max_possible_emd == 0:
        return 1.0
    normalized_similarity = 1.0 - (emd / max_possible_emd)
    return max(0.0, min(1.0, normalized_similarity))


def compute_combined_metric_euclidean(norm_emd, bc):
    """
    Combined metric using Euclidean distance: sqrt(norm_emd^2 + bc^2)

    Parameters:
    - norm_emd: Normalized EMD similarity (0-1, where 1 is best)
    - bc: Bhattacharyya coefficient (0-1, where 1 is best)

    Returns:
    - Combined score between 0 and sqrt(2) (higher is better)
    """
    combined = np.sqrt(norm_emd ** 2 + bc ** 2)
    return combined


def compute_combined_metric_average(norm_emd, bc):
    """
    Combined metric using simple average: (norm_emd/2) + (bc/2)

    Parameters:
    - norm_emd: Normalized EMD similarity (0-1, where 1 is best)
    - bc: Bhattacharyya coefficient (0-1, where 1 is best)

    Returns:
    - Combined score between 0 and 1 (higher is better)
    """
    combined = (norm_emd / 2.0) + (bc / 2.0)
    return combined


def split_histogram_balanced_random_equal(ratings_count, seed=None, balance_factor=0.47):
    """
    Split one image's ratings into two groups of equal size (the two groups add up to the original histogram).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    total_votes = int(sum(ratings_count))
    target_size = total_votes // 2

    hist1 = np.zeros(len(ratings_count), dtype=int)
    hist2 = np.zeros(len(ratings_count), dtype=int)

    # Create list of bins with votes
    bins_with_votes = [(i, int(count)) for i, count in enumerate(ratings_count) if count > 0]

    if len(bins_with_votes) == 0:
        return hist1, hist2

    # Shuffle to randomize processing order
    random.shuffle(bins_with_votes)

    group1_current = 0

    # Process all bins except the last one
    for idx, (bin_idx, count) in enumerate(bins_with_votes[:-1]):
        group1_remaining = target_size - group1_current
        group2_current = int(hist2.sum())
        group2_remaining = target_size - group2_current

        # Calculate votes remaining after this bin
        votes_remaining_after = sum(c for j, c in bins_with_votes[idx + 1:])

        if group1_remaining == 0:
            # Group1 is full
            hist2[bin_idx] = count
        elif group2_remaining == 0:
            # Group2 is full
            hist1[bin_idx] = count
        else:
            # both groups still need ratings: draw the share of this rating value at random

            # Calculate hard constraints
            min_to_group1 = max(0, group1_remaining - votes_remaining_after)
            max_to_group1 = min(count, group1_remaining, count - max(0, group2_remaining - votes_remaining_after))

            # Ensure valid range
            min_to_group1 = max(0, min_to_group1)
            max_to_group1 = min(count, max_to_group1)

            if min_to_group1 > max_to_group1:
                # Constraints conflict - take what we must
                to_group1 = min(group1_remaining, count)
            elif min_to_group1 == max_to_group1:
                # Only one option
                to_group1 = min_to_group1
            else:
                # random share within 50% +/- balance_factor
                ideal_split = count / 2.0

                # Define allowed deviation from 50-50 based on balance_factor
                max_deviation = count * balance_factor

                # Calculate balanced range
                balanced_min = max(min_to_group1, int(ideal_split - max_deviation))
                balanced_max = min(max_to_group1, int(ideal_split + max_deviation))

                # Ensure valid range
                balanced_min = max(balanced_min, min_to_group1)
                balanced_max = min(balanced_max, max_to_group1)

                if balanced_min > balanced_max:
                    # the equal-size constraint leaves no value inside 50% +/- balance_factor: use the feasible range
                    to_group1 = np.random.randint(min_to_group1, max_to_group1 + 1)
                else:
                    # use the restricted range
                    to_group1 = np.random.randint(balanced_min, balanced_max + 1)

            hist1[bin_idx] = to_group1
            hist2[bin_idx] = count - to_group1
            group1_current += to_group1

    # Handle the last bin - it must fill remaining space exactly
    if len(bins_with_votes) > 0:
        last_bin_idx, last_count = bins_with_votes[-1]
        group1_remaining = target_size - group1_current

        # Last bin must provide exactly what's needed
        to_group1 = min(group1_remaining, last_count)
        hist1[last_bin_idx] = to_group1
        hist2[last_bin_idx] = last_count - to_group1

    # Validation
    total_group1 = int(hist1.sum())
    total_group2 = int(hist2.sum())

    assert total_group1 == target_size or total_group1 == target_size + 1,\
        f"Group1 size mismatch: {total_group1} not in [{target_size}, {target_size + 1}]"
    assert total_group1 + total_group2 == total_votes,\
        f"Total mismatch: {total_group1 + total_group2} != {total_votes}"
    assert abs(total_group1 - total_group2) <= 1,\
        f"Groups differ by more than 1: {total_group1} vs {total_group2}"

    return hist1, hist2


def generate_unique_splits(ratings_count, K=300, balance_factor=0.47):
    """
    Generate K unique balanced random splits with equal group sizes.
    Ensures no duplicate splits by checking uniqueness.

    Parameters:
    - ratings_count: histogram of vote counts
    - K: number of unique splits to generate
    """
    splits = []
    seeds_used = []
    seen_splits = set()

    attempts = 0
    max_attempts = K * 100
    k = 0

    while k < K and attempts < max_attempts:
        # zlib.crc32 instead of hash(): Python's hash() of a string changes between runs, crc32 does not
        seed = k * 1000 + attempts + zlib.crc32(str(ratings_count.tolist()).encode()) % 10000

        try:
            h1, h2 = split_histogram_balanced_random_equal(ratings_count, seed=seed, balance_factor=balance_factor)

            # Create unique identifier
            split_id = tuple(sorted([tuple(h1.tolist()), tuple(h2.tolist())]))

            if split_id not in seen_splits:
                splits.append((h1, h2))
                seeds_used.append(seed)
                seen_splits.add(split_id)
                k += 1
        except Exception as e:
            pass

        attempts += 1

    if len(splits) < K:
        print(
            f"\nWarning: Could only generate {len(splits)} unique splits out of {K} requested after {attempts} attempts")
        print(f"This may indicate limited variation possible for this histogram")

    return splits, seeds_used


def analyze_image_detailed(image_id, human_hist, llm_hist, K=300, output_dir='image_results',
                           challenge_id=None, total_votes=None, llms=None, caption=None,
                           challenge_name=None, challenge_description=None, bin_value=None):
    """
    Analyze a single image with detailed history of all splits

    Returns dictionary with summary results and saves detailed CSV
    """
    human_hist = parse_histogram(human_hist)
    llm_hist = parse_histogram(llm_hist)

    os.makedirs(output_dir, exist_ok=True)

    # Calculate maximum possible EMD for normalization (number of bins - 1)
    num_bins = len(human_hist)
    max_possible_emd = num_bins - 1

    detailed_results = []

    # Generate K unique splits
    splits, seeds_used = generate_unique_splits(human_hist, K)

    actual_K = len(splits)
    if actual_K < K:
        print(f"\n  Image {image_id}: Generated {actual_K}/{K} unique splits")

    for k in range(len(splits)):
        h1, h2 = splits[k]

        emd = compute_emd(h1, h2)
        bc = compute_bhattacharyya(h1, h2)

        # Calculate normalized EMD (similarity score: 1 = identical, 0 = worst)
        norm_emd = normalize_emd_to_similarity(emd, max_possible_emd)

        # Calculate combined metrics
        combined_euclidean = compute_combined_metric_euclidean(norm_emd, bc)
        combined_average = compute_combined_metric_average(norm_emd, bc)

        detailed_results.append({
            'split_number': k + 1,
            'split_seed': seeds_used[k],
            'human_group1_hist': str(h1.tolist()),
            'human_group2_hist': str(h2.tolist()),
            'human_human_emd': emd,
            'human_human_norm_emd': norm_emd,
            'human_human_bc': bc,
            'human_human_combined_euclidean': combined_euclidean,
            'human_human_combined_average': combined_average,
            'group1_total': h1.sum(),
            'group2_total': h2.sum(),
            'total_votes': h1.sum() + h2.sum(),
            'split_difference': abs(h1.sum() - h2.sum())
        })

    detailed_df = pd.DataFrame(detailed_results)

    # Add metadata
    detailed_df.insert(0, 'Image ID', image_id)
    detailed_df.insert(1, 'Challenge ID', challenge_id if challenge_id is not None else '')
    detailed_df.insert(2, 'Total Votes', total_votes if total_votes is not None else '')
    detailed_df.insert(3, 'Caption', caption if caption is not None else '')
    detailed_df.insert(4, 'Challenge Name', challenge_name if challenge_name is not None else '')
    detailed_df.insert(5, 'Challenge Description', challenge_description if challenge_description is not None else '')
    detailed_df.insert(6, 'Bin', bin_value if bin_value is not None else '')

    # Compute LLM vs Human metrics
    llm_human_emd = compute_emd(llm_hist, human_hist)
    llm_human_norm_emd = normalize_emd_to_similarity(llm_human_emd, max_possible_emd)
    llm_human_bc = compute_bhattacharyya(llm_hist, human_hist)
    llm_human_combined_euclidean = compute_combined_metric_euclidean(llm_human_norm_emd, llm_human_bc)
    llm_human_combined_average = compute_combined_metric_average(llm_human_norm_emd, llm_human_bc)

    detailed_df['llm_human_emd'] = llm_human_emd
    detailed_df['llm_human_norm_emd'] = llm_human_norm_emd
    detailed_df['llm_human_bc'] = llm_human_bc
    detailed_df['llm_human_combined_euclidean'] = llm_human_combined_euclidean
    detailed_df['llm_human_combined_average'] = llm_human_combined_average

    detailed_df['original_human_hist'] = str(human_hist.tolist())
    detailed_df['original_llm_hist'] = str(llm_hist.tolist())

    # EMD (lower is better, so >= means the split is worse or equal) and BC (higher is better)
    detailed_df['llm_emd_better_than_split'] = detailed_df['human_human_emd'] >= llm_human_emd
    detailed_df['llm_bc_better_than_split'] = detailed_df['human_human_bc'] <= llm_human_bc

    # normalized EMD (higher is better)
    detailed_df['llm_norm_emd_better_than_split'] = detailed_df['human_human_norm_emd'] <= llm_human_norm_emd

    # Combined metrics comparisons (higher is better for both combined metrics)
    detailed_df['llm_combined_euclidean_better_than_split'] = detailed_df[
                                                                  'human_human_combined_euclidean'] <= llm_human_combined_euclidean
    detailed_df['llm_combined_average_better_than_split'] = detailed_df[
                                                                'human_human_combined_average'] <= llm_human_combined_average

    # Cumulative p-values for all metrics
    detailed_df['cumulative_p_value_emd'] = detailed_df['llm_emd_better_than_split'].cumsum() / detailed_df[
        'split_number']
    detailed_df['cumulative_p_value_norm_emd'] = detailed_df['llm_norm_emd_better_than_split'].cumsum() / detailed_df[
        'split_number']
    detailed_df['cumulative_p_value_bc'] = detailed_df['llm_bc_better_than_split'].cumsum() / detailed_df[
        'split_number']
    detailed_df['cumulative_p_value_combined_euclidean'] = detailed_df[
                                                               'llm_combined_euclidean_better_than_split'].cumsum() /\
                                                           detailed_df['split_number']
    detailed_df['cumulative_p_value_combined_average'] = detailed_df[
                                                             'llm_combined_average_better_than_split'].cumsum() /\
                                                         detailed_df['split_number']

    # Save detailed results
    output_file = os.path.join(output_dir, f'{image_id}.csv')
    detailed_df.to_csv(output_file, index=False)

    # Calculate summary statistics
    human_human_emds = detailed_df['human_human_emd'].values
    human_human_norm_emds = detailed_df['human_human_norm_emd'].values
    human_human_bcs = detailed_df['human_human_bc'].values
    human_human_combined_euclideans = detailed_df['human_human_combined_euclidean'].values
    human_human_combined_averages = detailed_df['human_human_combined_average'].values

    actual_K = len(splits)

    emd_better_count = np.sum(human_human_emds >= llm_human_emd)
    norm_emd_better_count = np.sum(human_human_norm_emds <= llm_human_norm_emd)
    bc_better_count = np.sum(human_human_bcs <= llm_human_bc)
    combined_euclidean_better_count = np.sum(human_human_combined_euclideans <= llm_human_combined_euclidean)
    combined_average_better_count = np.sum(human_human_combined_averages <= llm_human_combined_average)

    p_value_emd = emd_better_count / actual_K
    p_value_norm_emd = norm_emd_better_count / actual_K
    p_value_bc = bc_better_count / actual_K
    p_value_combined_euclidean = combined_euclidean_better_count / actual_K
    p_value_combined_average = combined_average_better_count / actual_K

    split_differences = detailed_df['split_difference'].values
    total_votes_arr = detailed_df['total_votes'].values

    summary = {
        'image_id': image_id,
        'actual_K': actual_K,
        'requested_K': K,
        'llm_human_emd': llm_human_emd,
        'llm_human_norm_emd': llm_human_norm_emd,
        'llm_human_bc': llm_human_bc,
        'llm_human_combined_euclidean': llm_human_combined_euclidean,
        'llm_human_combined_average': llm_human_combined_average,
        'human_human_emd_mean': np.mean(human_human_emds),
        'human_human_emd_std': np.std(human_human_emds),
        'human_human_emd_min': np.min(human_human_emds),
        'human_human_emd_max': np.max(human_human_emds),
        'human_human_emd_median': np.median(human_human_emds),
        'human_human_norm_emd_mean': np.mean(human_human_norm_emds),
        'human_human_norm_emd_std': np.std(human_human_norm_emds),
        'human_human_norm_emd_min': np.min(human_human_norm_emds),
        'human_human_norm_emd_max': np.max(human_human_norm_emds),
        'human_human_norm_emd_median': np.median(human_human_norm_emds),
        'human_human_bc_mean': np.mean(human_human_bcs),
        'human_human_bc_std': np.std(human_human_bcs),
        'human_human_bc_min': np.min(human_human_bcs),
        'human_human_bc_max': np.max(human_human_bcs),
        'human_human_bc_median': np.median(human_human_bcs),
        'human_human_combined_euclidean_mean': np.mean(human_human_combined_euclideans),
        'human_human_combined_euclidean_std': np.std(human_human_combined_euclideans),
        'human_human_combined_euclidean_min': np.min(human_human_combined_euclideans),
        'human_human_combined_euclidean_max': np.max(human_human_combined_euclideans),
        'human_human_combined_euclidean_median': np.median(human_human_combined_euclideans),
        'human_human_combined_average_mean': np.mean(human_human_combined_averages),
        'human_human_combined_average_std': np.std(human_human_combined_averages),
        'human_human_combined_average_min': np.min(human_human_combined_averages),
        'human_human_combined_average_max': np.max(human_human_combined_averages),
        'human_human_combined_average_median': np.median(human_human_combined_averages),
        'emd_better_count': emd_better_count,
        'norm_emd_better_count': norm_emd_better_count,
        'bc_better_count': bc_better_count,
        'combined_euclidean_better_count': combined_euclidean_better_count,
        'combined_average_better_count': combined_average_better_count,
        'p_value_emd': p_value_emd,
        'p_value_norm_emd': p_value_norm_emd,
        'p_value_bc': p_value_bc,
        'p_value_combined_euclidean': p_value_combined_euclidean,
        'p_value_combined_average': p_value_combined_average,
        'is_close_enough_emd': p_value_emd > 0.05,
        'is_close_enough_norm_emd': p_value_norm_emd > 0.05,
        'is_close_enough_bc': p_value_bc > 0.05,
        'is_close_enough_combined_euclidean': p_value_combined_euclidean > 0.05,
        'is_close_enough_combined_average': p_value_combined_average > 0.05,
        'split_difference_mean': np.mean(split_differences),
        'split_difference_max': np.max(split_differences),
        'total_votes': total_votes_arr[0] if len(total_votes_arr) > 0 else 0,
        'detailed_csv_file': output_file
    }

    return summary


def main(input_file, summary_output_file, detailed_output_dir, K):
    """
    Main function to process all images with detailed tracking

    Parameters:
    - input_file: path to Excel file with columns [Image ID, Human-ratings, LLM-ratings]
    - summary_output_file: path to save summary results CSV
    - detailed_output_dir: directory to save individual image CSV files
    - K: number of resampling iterations (default 300)
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file)

    expected_columns = ['Image ID', 'AVA-Ratings', 'LLM-Rating']
    if not all(col in df.columns for col in expected_columns):
        print(f"Warning: Expected columns {expected_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        df.columns = df.columns.str.strip()

    os.makedirs(detailed_output_dir, exist_ok=True)

    summary_results = []

    print(f"\nProcessing {len(df)} images with K={K} resamples each...")
    print("Split rule: equal-size groups; share of each rating value in group 1 drawn within 3-97%")
    print(f"Detailed results will be saved to: {detailed_output_dir}/")
    print(f"Each image will have its own CSV file named by Image ID\n")
    print(f"Metrics:")
    print(f"  - Normalized EMD (0-1, where 1=identical, 0=worst overlap)")
    print(f"  - Combined Euclidean: sqrt(norm_emd^2 + bc^2)")
    print(f"  - Combined Average: (norm_emd/2) + (bc/2)")
    print(f"  - TRUE/FALSE comparisons for all metrics\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            summary = analyze_image_detailed(
                image_id=row['Image ID'],
                human_hist=row['AVA-Ratings'],
                llm_hist=row['LLM-Rating'],
                K=K,
                output_dir=detailed_output_dir,
                challenge_id=row.get('Challenge ID', None),
                total_votes=row.get('Total Votes', None),
                caption=row.get('Caption', None),
                challenge_description=row.get('Challenge Description', None),
                bin_value=row.get('Bin', None),
            )
            summary_results.append(summary)
        except Exception as e:
            print(f"\nError processing image {row['Image ID']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(summary_output_file, index=False)

    print(f"\n{'=' * 70}")
    print(f"Summary results saved to: {summary_output_file}")
    print(f"Detailed results saved to: {detailed_output_dir}/ (one CSV per image)")
    print(f"{'=' * 70}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total images analyzed: {len(summary_df)}")
    print(f"Total detailed CSV files created: {len(summary_df)}")

    if 'actual_K' in summary_df.columns:
        avg_unique = summary_df['actual_K'].mean()
        min_unique = summary_df['actual_K'].min()
        images_with_fewer = (summary_df['actual_K'] < summary_df['requested_K']).sum()
        print(f"\nUnique Splits Generation:")
        print(f"  Average unique splits per image: {avg_unique:.1f}")
        print(f"  Minimum unique splits for any image: {min_unique}")
        print(f"  Images with fewer than requested K: {images_with_fewer}")

    print(f"\nSplit Statistics:")
    print(f"  Average split difference: {summary_df['split_difference_mean'].mean():.2f}")
    print(f"  Maximum split difference: {summary_df['split_difference_max'].max()}")

    print(f"\nEMD Analysis:")
    print(
        f"  Images where LLM is 'close enough' (p > 0.05): {summary_df['is_close_enough_emd'].sum()} ({summary_df['is_close_enough_emd'].sum() / len(summary_df) * 100:.1f}%)")
    print(f"  Mean LLM-Human EMD: {summary_df['llm_human_emd'].mean():.4f}")
    print(f"  Mean Human-Human EMD: {summary_df['human_human_emd_mean'].mean():.4f}")
    print(f"  Median p-value: {summary_df['p_value_emd'].median():.4f}")

    print(f"\nNormalized EMD Analysis:")
    print(
        f"  Images where LLM is 'close enough' (p > 0.05): {summary_df['is_close_enough_norm_emd'].sum()} ({summary_df['is_close_enough_norm_emd'].sum() / len(summary_df) * 100:.1f}%)")
    print(f"  Mean LLM-Human Normalized EMD: {summary_df['llm_human_norm_emd'].mean():.4f}")
    print(f"  Mean Human-Human Normalized EMD: {summary_df['human_human_norm_emd_mean'].mean():.4f}")
    print(f"  Median p-value: {summary_df['p_value_norm_emd'].median():.4f}")

    print(f"\nBhattacharyya Coefficient Analysis:")
    print(
        f"  Images where LLM is 'close enough' (p > 0.05): {summary_df['is_close_enough_bc'].sum()} ({summary_df['is_close_enough_bc'].sum() / len(summary_df) * 100:.1f}%)")
    print(f"  Mean LLM-Human BC: {summary_df['llm_human_bc'].mean():.4f}")
    print(f"  Mean Human-Human BC: {summary_df['human_human_bc_mean'].mean():.4f}")
    print(f"  Median p-value: {summary_df['p_value_bc'].median():.4f}")

    print(f"\nCombined Euclidean Metric Analysis:")
    print(
        f"  Images where LLM is 'close enough' (p > 0.05): {summary_df['is_close_enough_combined_euclidean'].sum()} ({summary_df['is_close_enough_combined_euclidean'].sum() / len(summary_df) * 100:.1f}%)")
    print(f"  Mean LLM-Human Combined Euclidean: {summary_df['llm_human_combined_euclidean'].mean():.4f}")
    print(f"  Mean Human-Human Combined Euclidean: {summary_df['human_human_combined_euclidean_mean'].mean():.4f}")
    print(f"  Median p-value: {summary_df['p_value_combined_euclidean'].median():.4f}")

    print(f"\nCombined Average Metric Analysis:")
    print(
        f"  Images where LLM is 'close enough' (p > 0.05): {summary_df['is_close_enough_combined_average'].sum()} ({summary_df['is_close_enough_combined_average'].sum() / len(summary_df) * 100:.1f}%)")
    print(f"  Mean LLM-Human Combined Average: {summary_df['llm_human_combined_average'].mean():.4f}")
    print(f"  Mean Human-Human Combined Average: {summary_df['human_human_combined_average_mean'].mean():.4f}")
    print(f"  Median p-value: {summary_df['p_value_combined_average'].median():.4f}")

    print(f"\nP-value Distribution (EMD):")
    print(f"  p < 0.01: {(summary_df['p_value_emd'] < 0.01).sum()} images")
    print(
        f"  0.01 ≤ p < 0.05: {((summary_df['p_value_emd'] >= 0.01) & (summary_df['p_value_emd'] < 0.05)).sum()} images")
    print(
        f"  0.05 ≤ p < 0.10: {((summary_df['p_value_emd'] >= 0.05) & (summary_df['p_value_emd'] < 0.10)).sum()} images")
    print(f"  p >= 0.10: {(summary_df['p_value_emd'] >= 0.10).sum()} images")

    print("\n" + "=" * 70)
    print("Example: Check detailed results for any image in the directory:")
    print(f"  {detailed_output_dir}/<Image_ID>.csv")
    print("Split rule: equal-size groups; share of each rating value in group 1 drawn within 3-97%")
    print(f"\nColumns in detailed CSVs:")
    print(f"  - human_human_norm_emd: Normalized EMD (0-1, 1=best)")
    print(f"  - human_human_combined_euclidean: Combined metric using Euclidean formula")
    print(f"  - human_human_combined_average: Combined metric using average formula")
    print(f"  - llm_*_better_than_split: TRUE/FALSE for each metric")
    print(f"  - cumulative_p_value_*: Running p-values for all metrics")
    print("=" * 70)

    return summary_df


if __name__ == "__main__":
    # Configuration
    for K in range(1000, 1001, 200):
        INPUT_FILE = r"results/combined-human-llm-rating-AVA.xlsx"   # from 0_build_alignment_table_AVA.py

        output_base = fr"results/split_half/AVA/{K}-iterations"
        os.makedirs(output_base, exist_ok=True)

        SUMMARY_OUTPUT_FILE = os.path.join(output_base, "summary_results.csv")
        DETAILED_OUTPUT_DIR = os.path.join(output_base, "image_results")
        LOG_FILE = os.path.join(output_base, "summary.txt")

        logger = Logger(LOG_FILE)
        sys.stdout = logger

        print("=" * 70)
        print("HUMAN-LLM RATING ANALYSIS - AVA, SPLIT-HALF BASELINE")
        print("WITH NORMALIZED EMD AND COMBINED METRICS")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input file: {INPUT_FILE}")
        print(f"K (resamples): {K}")
        print(f"Output directory: {output_base}")
        print(f"Log file: {LOG_FILE}")
        print("Split rule: equal-size groups; share of each rating value in group 1 drawn within 3-97%")
        print("\nMetrics:")
        print("  - Normalized EMD (0-1 scale, 1=identical overlap)")
        print("  - Combined Euclidean: sqrt(norm_emd^2 + bc^2)")
        print("  - Combined Average: (norm_emd/2) + (bc/2)")
        print("  - TRUE/FALSE comparisons for all 5 metrics")
        print("=" * 70)
        print()

        try:
            results = main(INPUT_FILE, SUMMARY_OUTPUT_FILE, DETAILED_OUTPUT_DIR, K)

            print("\n" + "=" * 70)
            print("Analysis complete!")
            print("=" * 70)
            print(f"Summary: {SUMMARY_OUTPUT_FILE}")
            print(f"Per-image splits: {DETAILED_OUTPUT_DIR}/<Image_ID>.csv")
            print("=" * 70)
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

        except Exception as e:
            print(f"\n{'=' * 70}")
            print(f"ERROR: Analysis failed!")
            print(f"{'=' * 70}")
            print(f"Error message: {str(e)}")
            import traceback

            traceback.print_exc()
            print("=" * 70)

        finally:
            logger.close()
            sys.stdout = logger.terminal
            print(f"\nAll terminal output has been saved to: {LOG_FILE}")