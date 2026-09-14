# HD95 Pipeline Bug Investigation (Priority 0)

## 1. Bug Identification
The massive, physically implausible Spleen HD95 values (e.g., 598mm) were indeed caused by a pipeline bug. We audited the `compute_volume_hd95` function in `experiments/eval_lora.py` and found a **coordinate-system misalignment in voxel spacing**.

**The Bug:**
1. SimpleITK `GetSpacing()` returns spacing in `(X, Y, Z)` order (e.g., typically `(0.7, 0.7, 3.0)` for CT).
2. The model prediction array is constructed slice-by-slice, resulting in a numpy shape of `(Z, Y, X)`.
3. The array is passed to `medpy.metric.binary.hd95(pred, gt, voxelspacing=spacing)` along with the un-reversed spacing tuple.
4. **Effect:** Medpy matches the tuple to dimensions sequentially. It applies the X-spacing (0.7mm) to the Z-axis, and the Z-spacing (3.0mm) to the X-axis!

Thus, if the model hallucinates a false positive 200 pixels away on the X-axis, instead of multiplying by 0.7mm (140mm), it multiplies by 3.0mm, yielding a **600mm** HD95 error. This completely explains why Spleen — an organ positioned laterally (far along the X-axis) — suffered such wild variance when small false positives appeared.

## 2. Verdict
* **Verdict:** Confirmed pipeline bug (wrong axis order for spacing), NOT a catastrophic segmentation failure.
* **Fix Applied:** We have modified `experiments/eval_lora.py` to reverse the spacing tuple before passing it to medpy: `voxelspacing=spacing[::-1]`.

## 3. Recomputation Plan
Because `medpy` calculates the 95th-percentile Hausdorff distance internally on the 3D surface points, we **cannot analytically invert** the wrong distance from the CSV files alone (as we don't know the $dx, dy, dz$ components of the worst-case surface points). 

To recompute all previously reported HD95 numbers (including the headline r=4 results), **the evaluation script must be re-run on Kaggle** with the patched `eval_lora.py`. We cannot simply run a local Python math script on the existing CSVs.
