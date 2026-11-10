# Aesthetics with and without semantics: How semantic availability shapes aesthetic judgments

Code for the paper "Aesthetics with and without semantics: How semantic availability shapes aesthetic judgments"
by Arslan Javed, Bogdan Raducanu, Olivier Penacchio, Xim Cerda-Company, Fernando Vilariño and C. Alejandro Parraga.

The study compares two routes for predicting human aesthetic ratings on the AVA and MSC datasets:
a language-mediated route (GPT-4o captions, validated by image regeneration, rated by GPT-4o) and a
perceptual route (quantitative image properties, QIP, classified with an SVM).

## Repository structure

| Folder | Content |
|---|---|
| `01_dataset_preparation/` | AVA subset: truncated-Gaussian mean per image, all images of bins 1-3 and 9-10, 1,860 random images from each of bins 4-8 |
| `02_captions/` | GPT-4o captions of the AVA subset (`caption_AVA.py`) and of MSC (`caption_MSC.py`) |
| `03_caption_validation/` | Image regeneration (FLUX.1-schnell, SANA 1600M, Lumina-Next-SFT), CLIP cosine similarity and acceptance (threshold 0.62); second captions, replacement images and the final AVA subset |
| `04_human_similarity_experiment/` | 2AFC similarity experiment (PsychoPy/PsychoJS, run on Pavlovia) and the sigmoid fits that give the 0.62 threshold |
| `05_llm_rating/` | Aesthetic ratings of the captions with GPT-4o (`rate_captions_AVA.py`, `rate_captions_MSC.py`) |
| `06_alignment/` | Human split-half baseline, LLM-human alignment per image, bin-wise statistics and the AVA-MSC comparison |
| `07_perceptual/` | QIP extraction (Aesthetics Toolbox), class labels and the SVM classifier |

Scripts with a number are run in that order within their folder. The 0.62 threshold used in `03_caption_validation`
comes from the 2AFC experiment in `04_human_similarity_experiment`, whose image pairs were made with the regeneration
and CLIP scripts of `03_caption_validation`.

## Scripts behind the figures and tables

| Paper | Script |
|---|---|
| Datasets, Supplementary Figure S1 and Table 2 | `01_dataset_preparation/1_` to `3_`, `03_caption_validation/9_build_final_subset_AVA.py` |
| Caption validation, Supplementary S4 and Table 1 | `03_caption_validation/3_compare_models.py`, `4_remaining_failed_images.py`, `8_select_replacements.py` |
| CLIP threshold (2AFC), Supplementary Figure S2 | `04_human_similarity_experiment/analysis/1_preprocess_pavlovia.py`, `2_fit_sigmoid_thresholds.py`, `3_threshold_summary.py` |
| Split-half baseline (K = 1000) | `06_alignment/1_split_half_AVA.py`, `1_split_half_MSC.py` |
| Figure 3a, AVA bin-wise statistics | `06_alignment/2_binwise_statistics_AVA.py` |
| Figure 3b, MSC bin-wise statistics | `06_alignment/2_binwise_statistics_MSC.py` |
| Overall AVA-MSC comparison | `06_alignment/3_overall_AVA_vs_MSC.py` |
| Figure 4, Supplementary Figure S3 | `07_perceptual/5_classifier.py` (`N_CLASSES = 5` or `3`) |

## Pipeline

1. `01_dataset_preparation`: bins of all AVA images and the initial AVA subset.
2. `02_captions`: captions of the AVA subset and of MSC. MSC captions are not validated (the 0.62 threshold was
   calibrated on AVA pairs); MSC images without a caption are not used in the language-mediated analyses.
3. `03_caption_validation` (AVA). Steps 1-3 are run once per validation round (set `ROUND` and the caption file):
   * round 1, all first captions: `1_generate_images_*.py`, `2_clip_similarity.py`, `3_compare_models.py`.
     A caption is accepted when at least one of the three regenerated images reaches 0.62.
   * rounds 2 and 3, images that failed round 1, same captions: steps 1-3 again, then `4_remaining_failed_images.py`.
   * round 4, second caption: `5_copy_images_for_second_caption_AVA.py`, `6_caption_AVA_second.py`, steps 1-3.
   * round 5, replacements: `7_replacement_candidates_AVA.py` (once per bin), `02_captions/caption_AVA.py` on the
     candidates, steps 1-3, then `8_select_replacements.py`.
   * `9_build_final_subset_AVA.py`: final images and captions (`AVA_subset_images.csv`, `AVA_captions.csv`).
4. `05_llm_rating`: rate the captions with GPT-4o. With `NUM_RATINGS = None` each image receives as many LLM ratings
   as it has human ratings. Both scripts read the image and caption tables (data repository or `9_build_final_subset_AVA.py`).
5. `06_alignment`: `0_build_alignment_table_AVA.py` (and `0_build_alignment_table_MSC.py` when starting from the data
   repository), then the split-half scripts, then the statistics scripts.
6. `07_perceptual`: QIP extraction, class labels, matching to the AVA subset, classifier.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...     # needed only for 02_captions and 05_llm_rating
```

* Paths in the scripts are relative. Every input is either in `data/` or written to `results/` by an earlier script:
  * `data/AVA/`: `AVA.txt` (rating counts and challenge ID of every AVA image), `challenges.txt` and `images/`.
    These are part of the AVA dataset (Murray et al., 2012), which is not redistributed here; it is available, e.g., at
    https://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460
  * `data/MSC/images/`: MSC images in folders 1-5 (Penacchio et al., 2026)
  * `data/01_image_sets/` ... `data/07_QIP_features_and_classifier/`: the folders of the data repository
* Starting from the data repository, the analyses run without the images or the API: LLM-human alignment
  (`06_alignment`, starting with the two `0_build_alignment_table_` scripts), the 2AFC threshold (`04_.../analysis/2_` and `3_`)
  and the classifier (`07_perceptual/5_classifier.py` with the QIP tables of the data repository).
* The QIP scripts in `07_perceptual` (`1_` and `2_`) are batch scripts for the Aesthetics Toolbox
  (Redies et al., 2025, https://github.com/RBartho/Aesthetics-Toolbox) and are run from the toolbox's root folder.
  Both `1_` scripts compute the same 45 QIPs: `1_QIP_extraction.py` for MSC, and `1_QIP_extraction_parallel.py` for AVA
  (~255,000 images, in chunks merged with `2_merge_QIP_chunks.py`).
* `03_caption_validation/2_clip_similarity.py` uses SpLiCE (https://github.com/AI4LIFE-GROUP/SpLiCE) with `open_clip:ViT-B-32`.
* Image generation needs a GPU. Image generation and GPT-4o outputs are not deterministic; the captions, CLIP scores
  and LLM ratings used in the paper are in the data repository.

## Data

AVA (Murray et al., 2012) and MSC (Penacchio et al., 2026) images are not redistributed. Image IDs, captions,
CLIP validation scores, LLM rating distributions, alignment results and QIP feature tables are available at: [DATA DOI].

## Citation

[Add citation when published.]

## License

[Add license.]
