# Data Download Instructions

**IMPORTANT**: Do not commit any datasets to the Git repository. The `.gitignore` prevents tracking of the `data/` folder contents, but always double check.

## 1. Abd-CT (SABS)
1. Navigate to https://www.synapse.org/#!Synapse:syn3193805
2. Register for a Synapse account and sign the data use agreement to get access.
3. Download the SABS training data.
4. Run the SSL-ALPNet preprocessing script for CT (which windows HU to [-125, 275] and normalizes to [0, 1]).
5. Place the output into `data/SABS/sabs_CT_normalized/`.

## 2. Abd-MRI (CHAOS T2-SPIR)
1. Navigate to https://chaos.grand-challenge.org/
2. Register for a Grand-Challenge account and join the CHAOS challenge.
3. Download the MR data, specifically focusing on the T2-SPIR sequence.
4. Run the SSL-ALPNet preprocessing script for MRI (z-score normalization per volume).
5. Place the output into `data/CHAOST2/chaos_MR_T2_normalized/`.

## 3. SSL-ALPNet Supervoxels
1. Use the scripts in `third_party/SSL_ALPNet/` to generate supervoxels for the few-shot task.
2. Place the generated supervoxels in `data/SABS/supervoxels_5000/` and `data/CHAOST2/supervoxels_5000/` respectively.
