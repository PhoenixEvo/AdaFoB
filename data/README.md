# data/

Raw and preprocessed datasets are NOT committed to git.

## Sources (follow FoB exactly)
- Abd-CT (SABS / Multi-Atlas): https://www.synapse.org/#!Synapse:syn3193805
- Abd-MRI (CHAOS T2-SPIR): https://chaos.grand-challenge.org/
- DRIVE (retinal vessels, secondary): https://drive.grand-challenge.org/
- BraTS2020 (secondary): https://www.med.upenn.edu/cbica/brats2020/

## Preprocessing
Follow Ouyang et al. (SSL-ALPNet) exactly, as FoB does:
https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
- CT: HU window [-125, 275], normalize to [0,1]
- MRI: z-score per volume
- Slices resized to 256x256
- 5-fold CV, Setting I and Setting II; 1-way 1-shot primary
