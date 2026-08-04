# FoB Code Audit

**a. Where is Np (number of background prompts) defined? Is it a config value or hard-coded?**
Np is hard-coded as `10`. It is not a config value.
Locations assuming fixed Np:
- `models/FoB.py`:L294 `self.num_points = 10` in `FewShotSeg.__init__`.
- `models/FoB.py`:L14 is hard-coded to `num_points=10` by default in `IDR.__init__` and uses it to define `offset_pred` output channels.
- `models/FoB.py`:L143-L146 `Head` uses `out_channels=10`, predicting exactly 10 heatmaps.
- `models/FoB.py`:L215 `MaskedAttention` defines `self.conv = nn.Conv2d(10, 1...)` which expects a 10-channel tensor.
- `models/FoB.py`:L217 `self.sin_pos = self.get_sinusoid_encoding_table(10, feature_dim)` hard-codes position embeddings to length 10.
- `models/FoB.py`:L420 `rac_loss += torch.clamp(0.5 + cos_sim, min=0) / self.num_points` uses it for loss normalization.
- `models/FoB.py`:L366,L394 The `for i in range(self.num_points):` loops assume exactly Np points.

**b. How is the "ring prior" implemented concretely?**
**(i) BPPC support prompt sampling:**
Implemented in `models/FoB.py`:L525-L530 via differential dilation band:
```python
    def get_ring(self, label, kernel_size=9):
        label_dilate_9 = self.dilate_label(label, kernel_size)
        label_dilate_5 = self.dilate_label(label, 15)
        ring = label_dilate_9 - label_dilate_5
        return ring
```
It is called in `models/FoB.py`:L546 with `mask = self.get_ring(mask, kernel_size=21)`.
Since padding for dilation is `kernel_size//2`, `kernel=21` provides a radius of 10, and `kernel=15` provides a radius of 7. This implements the differential band conceptually representing $\rho(M, r=15) - \rho(M, r-\epsilon=2)$ from Eq. 1.
**(ii) Ring-topology graph $A^{ring}$ in SPR:**
Implemented in `models/FoB.py`:L69-L75:
```python
    def build_ring_adj(self, K, B, device):
        A = torch.zeros(K, K, device=device)
        for i in range(K):
            A[i, (i - 1) % K] = 1
            A[i, (i + 1) % K] = 1
        A = A.unsqueeze(0).expand(B, -1, -1)
        return A
```

**c. Where do ground-truth query background prompts come from during training?**
They are generated dynamically during the forward pass in `models/FoB.py`:L405-L408:
```python
            if train:
                gt = self.uniform_sample_contour(qry_labels.unsqueeze(0).float(), num_keypoints=self.num_points) # [10,2]
                heatmaps_gt = self.generate_keypoint_heatmaps(img_size, gt) #(10, 256, 256)
```
This exact function call (`self.uniform_sample_contour`) is where GAP will later inject skeleton-based targets.

**d. What would break if Np varied per episode? Minimal Refactor Proposal:**
If `Np` varied per episode, the hard-coded loops and static model layers (e.g. `Head` outputting exactly 10 channels, `MaskedAttention` Conv2d expecting 10 channels, and `sin_pos` generating 10 embeddings) would crash because shapes would be dynamic or mismatched across the PyTorch graph.
**Minimal Refactor Proposal (Padding + Mask):**
1. Define a global maximum `MAX_NP = 10` (or greater). Let `Head` always predict `MAX_NP` channels.
2. Generate `sin_pos` for `MAX_NP`.
3. For episodes with actual $N_p < MAX_{NP}$, zero-pad the points/heatmaps up to `MAX_NP` and generate a boolean mask of shape `[B, MAX_NP]`.
4. Apply the mask to `sim` before convolution in `MaskedAttention`, and mask out the dummy background prompts when aggregating the `JointsMSELoss` and `rac_loss`.
This isolates the changes to a mask broadcast and avoids re-allocating layers or writing dynamic per-episode loops.

**e. Dataset Loaders:**
FoB uses `TrainDataset` and `TestDataset` in `dataloaders/datasets.py`.
It natively supports both **Abd-CT (SABS)** and **Abd-MRI (CHAOST2)** out of the box.
Evidence: `dataloaders/datasets.py`:L130-L135 directly checks for `args['dataset'] == 'CHAOST2'` and `args['dataset'] == 'SABS'`, loading from `chaos_MR_T2_normalized` and `sabs_CT_normalized` directories respectively.

**f. Pretrained Checkpoints:**
Huggingface repository `https://huggingface.co/PrimeBo1/FoB_SAM` contains:
- `exps_train_on_CHAOST2_FSMIS_FoB.zip`
- `exps_train_on_SABS_FSMIS_FoB.zip`
Both SABS and CHAOST2 datasets have their pretrained weights available, which include all necessary cross-validation folds. We will use these for all FoB baseline numbers.
