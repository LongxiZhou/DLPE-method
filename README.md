# Seeing the Unseen: Discovering Interpretable Sub-Visual Abnormalities in CT Scans of COVID-19 Patients and Survivors by Deep Learning
## Overview
This repository provides the predictive model described in the paper:
```
Longxi Zhou, et al. "Seeing the Unseen: Discovering Interpretable Sub-Visual Abnormalities in CT Scans of COVID-19 Patients and Survivors by Deep Learning"
```

## Description
Deep-LungParenchyma-Enhancing (DLPE) is a computer-aided detection (CADe) method for detecting and quantifying pulmonary parenchyma lesions on chest computerized tomography (CT). Using deep-learning, DLPE removes irrelevant tissues other than pulmonary parenchyma, and calculates the scan-level optimal window which enhances parenchyma lesions for dozens of times compared to the lung window. Aided by DLPE, radiologists discovered novel and interpretable lesions from COVID-19 inpatients and survivors, which are previously invisible under the original lung window and have strong predictive power for key COVID-19 clinical metrics and sequelae.
<div align="center">
  <img src="./resources/Fig_one.png" width="800" height="800">
</div>

## Run the model
- Step 1): Dowload the codes in github (note in github, file "trained_models" and "example_data" are empty files).
- Step 2): Download the file: "trained_models" and "example_data" from [Google Drive](https://drive.google.com/drive/folders/16ZvZfhqMmuF7wqNPKUOntw2P-Mfx5C4l?usp=sharing).
- Step 3): Replace the "trained_models" and "example_data" with Google Drive downloaded.
- Step 4): Establish the python environment by 'resources/req.txt'.
- Step 5): Open 'interface/dcm_to_enhanced.py', follow the instruction to change global parameter "trained_model_top_dict", "dcm_directory" and "enhance_array_output_directory".
- Step 6): Run 'interface/dcm_to_enhanced.py' and it will cost about one minute for each chest CT scan.

## Contact

If you request our training code for DLPE method, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa.

