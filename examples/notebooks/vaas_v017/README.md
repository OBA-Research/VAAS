# VAAS v0.1.7 – Colab Notebooks

This directory contains a curated set of Google Colab notebooks
demonstrating how to use VAAS v0.1.7 for image anomaly detection
and visual integrity analysis.

The notebooks are maintained as part of the official VAAS release
and reflect the intended usage and design philosophy of the framework.

All notebooks are:

- compatible with VAAS v0.1.7,
- inference-only,
- runnable in Google Colab,
- and based on publicly hosted example images.

---

## Getting Started

1. Open any notebook in Google Colab.
2. Run the installation cell.
3. Execute the notebook top to bottom.

No local setup is required.

---

## Notebook Index

### Practitioner Workflows

1. **01_detecting_image_manipulation_quick_start**  
   [Quick start for detecting visual inconsistency in a single image.](https://colab.research.google.com/drive/1tBZIMXjDLwjrbnHGNdtVgsyXoaQ2q6KK?usp=sharing)

2. **02_where_was_the_image_manipulated**  
   [Localization of suspicious regions using anomaly maps.](https://colab.research.google.com/drive/1EBZYx56DQcTaxPlP_hWCnXaVDzjcv_TV?usp=sharing)

3. **03_understanding_vaas_scores_sf_sp_sh**  
   [Explanation of VAAS scoring components and their interpretation.](https://colab.research.google.com/drive/1yNKrlwue9BItzqmhZUZ4-3d5kBAm9qys?usp=sharing)

4. **04_effect_of_alpha_on_anomaly_scoring**  
   [Impact of the alpha parameter on anomaly sensitivity.](https://colab.research.google.com/drive/1IlBhIOzUEqaeqJnPJ6bWfjw0nv6BBATe?usp=sharing)

5. **05_running_vaas_on_cpu_cuda_mps**  
   [Device selection and execution on different hardware backends.](https://colab.research.google.com/drive/1XeQjEdlWtisZoDDPp6WxwbNxoYC43wyk?usp=sharing)

6. **06_loading_vaas_models_from_huggingface**
   [Loading pretrained VAAS models and understanding repository structure.](https://colab.research.google.com/drive/16X5S_aarUKGktMYlW2bo2Fp4p5VX5p85?usp=sharing)

7. **07_batch_analysis_with_vaas_folder_workflow**  
   [Applying VAAS to multiple images using explicit Python workflows.](https://colab.research.google.com/drive/1RBoG70bH9k3YceU0VdyfewlrDgjOOaom?usp=sharing)

8. **08_ranking_images_by_visual_suspicion**  
   [Ranking image collections by anomaly score for triage.](https://colab.research.google.com/drive/18D4eV_fgomOIrxsyP_U__HYrTl-ZtC8e?usp=sharing)

---

### Research and Extension

9. **09_using_vaas_outputs_in_downstream_research**  
   [Reusing VAAS scores and anomaly maps in research pipelines.](https://colab.research.google.com/drive/1AiciR4GcXimFgr7M8Q8fXFCTekpmXN_X?usp=sharing)

10. **10_known_limitations_and_future_research_directions**  
    [Documented API boundaries and future research directions.](https://colab.research.google.com/drive/1Vr2ufQp-pWwMh6tQt84DilYu6ESm-ZP2?usp=sharing)

---

## Notes

Future VAAS releases may introduce new notebook sets
without modifying this directory.
