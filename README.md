# A Rapid, Accurate and Machine-agnostic Segmentation and Quantification Method for CT-scan-based COVID-19 Diagnostics
## Overview

<div align="center">
  <img src="./resources/main.png" width="800" height="450">
</div>
This repository provides the predictive model described in the paper:

```
Longxi Zhou, et al. "A Rapid, Accurate and Machine-agnostic Segmentation and Quantification Method for CT-based COVID-19 Diagnosis"
```

## Contents
- 01.introductory.demo contains and example prediction of our model
- 02.our.model contains our full-fledged model
- 03.baselines.demo contains code for baselines

The trained models of our model and the baseline methods are stored on [Google Drive](https://drive.google.com/drive/folders/1_-W8HcHpnBS_9Hkz6P5QfE6Gw-pNXxZ7?usp=sharing). Please respect the folder structure in the drive when downloading. 

### Example Data on the Google Drive:
The data for `02.our.model` is in in `02.our.model/patients/`. Our method will preprocessing these files, predict, and visualize the infection segmentations.
The data for models in `03.baselines.demo` is in `CT_scan_spatial_signal_normalized/`, which are same arrays with arrays stored in `./02.our.model/standard/patient_id/time_point/` after the preprocessing. Read the `readme` files for these comparisions for detailed information.
The `Lung_segmentation_mask/` stores the lung_masks for the scans: 1 means inside lungs, 0 means outside lungs. All methods used the same lung masks to exclude false-positives when we did the quatitative analysis.

## Contact

If you request our training code/simulation model for COVID-19, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa.

test
