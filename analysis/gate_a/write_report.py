"""Assemble the Gate A report only from executed artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def md(df):
 return df.to_markdown(index=False,floatfmt=".6f")

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();r=a.run_dir
 analysis=r/"corrected_analysis"
 facts=json.loads((analysis/"summary.json").read_text())
 measures=pd.read_csv(r/"metrics_full/corrected_measurements.csv")
 sums=pd.read_csv(r/"metrics_full/corrected_summary.csv")
 decomp=pd.read_csv(analysis/"decomposition_both_estimands.csv")
 replacements=pd.read_csv(analysis/"historical_field_replacements.csv")
 coord=pd.read_csv(r/"coordinate_diagnostic/coordinate_sensitivity.csv")
 cm=pd.read_csv(r/"coordinate_diagnostic/slice_metrics.csv")
 timing=json.loads((r/"coordinate_diagnostic/summary.json").read_text())["timing"]
 lock=json.loads((r/"protocol_lock/gate_a_protocol.json").read_text())
 lockcommit=json.loads((r/"protocol_lock_commit.json").read_text())
 env=json.loads(Path("audit/environment_gpu.json").read_text())
 status=pd.read_csv("audit/test_status_v2.csv")
 pytest=status[status.reason_and_consequence.str.contains("JUnit",na=False)]
 # Numeric parity table keeps relative denominators explicit.
 p10=json.loads((r/"parity_width10.json").read_text());p24=json.loads((r/"parity_width24.json").read_text())
 parity=[]
 for name,v in p10["comparisons"].items():
  w=p24["comparisons"][name]
  parity.append({"tensor":name,"width10 max abs":v["max_absolute"],"width10 max rel":v["max_relative"],"width24 max abs":w["max_absolute"],"width24 max rel":w["max_relative"],"width24": "PASS" if w["pass"] else "FAIL"})
 fixes=[]
 for stem in ["parity_width24_validring","parity_width24_trimmed"]:
  d=json.loads((r/(stem+".json")).read_text())
  fixes.append({"arm":d["single_factor_arm"],"coordinate max abs":d["comparisons"]["negative_xy"]["max_absolute"],
                "postprocessed logit max abs":d["comparisons"]["sam_R1_postprocessed_logits"]["max_absolute"],
                "binary changed pixels":d["comparisons"]["final_binary_mask"]["different_elements"],"parity":d["numerical_parity"]})
 primary=[]
 for (arm,post),g in measures.groupby(["arm","post"]):
  def s(metric):return sums[(sums.arm==arm)&(sums.post==post)&(sums.metric==metric)].iloc[0]
  ds,hs=s("dice"),s("surface_hd95")
  primary.append({"arm":arm.upper(),"post":post,"episode Dice %":ds.episode_weighted*100,"patient Dice %":ds.patient_balanced*100,
                  "episode 3D HD95 mm":hs.episode_weighted,"patient 3D HD95 mm":hs.patient_balanced,"median HD95":hs.episode_median,"p90 HD95":hs.episode_p90})
 part=[];direct=[]
 for arm in ["b0","b3b"]:
  for post in ["raw","cc"]:
   g=measures[(measures.arm==arm)&(measures.post==post)]
   part.append({"arm":arm.upper(),"post":post,**{c:g[c].mean() for c in ["fp","fn","detached_fp","other_annotated_organ_fp","residual_fp","removed_tp","removed_fp"]}})
   direct.append({"arm":arm.upper(),"post":post,**{c:g[c].mean() for c in ["legacy_surface_to_fg_pred95","directed_surface_pred_to_gt95","legacy_surface_to_fg_gt95","directed_surface_gt_to_pred95"]}})
 changes=[]
 for (arm,old),g in replacements.groupby(["arm","legacy_field"]):
  changes.append({"arm":arm.upper(),"old field":old,"replacement":g.replacement_field.iloc[0],"old episode mean":g.old_value.mean(),"new episode mean":g.new_value.mean(),"changed rows":int((g.change.abs()>1e-9).sum())})
 sensitivity=[]
 for eid,g in coord[coord.post.notna()].groupby("episode_id"):
  raw=g[g.post=="raw"].iloc[0];cc=g[g.post=="cc"].iloc[0]
  organ=cm[cm.episode_id==eid].iloc[0]
  sensitivity.append({"episode":eid,"patient":organ.patient_id,"organ":organ.organ,"raw Dice delta pp":raw.dice_change*100,"CC Dice delta pp":cc.dice_change*100,"CC 2D HD95 delta mm":cc.surface_hd95_change_mm,"CC changed pixels":cc.changed_pixels,"trigger":cc.materiality_trigger})
 conditions=pd.read_csv(analysis/"step5_csv_only_both_estimands.csv")
 wanted=["dice_k0_cc","dice_k10_cc","dice_k2_cc","dice_k4_cc","dice_k6_cc","dice_rand_k2_mean_cc","dice_rand_k4_mean_cc","dice_rand_k6_mean_cc","dice_prefix_oracle_cc","dice_greedy_oracle_cc"]
 s5table=conditions[conditions.metric.isin(wanted)].copy()
 s5table["episode_weighted"]*=100;s5table["patient_balanced"]*=100
 s5table=s5table[["metric","episode_weighted","patient_balanced"]]
 package_rows=[{"package":k,"version":v.get("version"),"module present":v.get("module_found")} for k,v in env["required"].items()]
 upper=2*timing["estimated_4009_slice_seconds_at_mean"]
 negative=coord[coord.kind=="negative"]
 text=f"""# Step 7 — Gate A resolution batch

Date: 2026-10-23. Starting and ending main-repository HEAD: 063192a33a1d096758ae50ff03d982d075355835.

**Gate A status: FAIL.**

**Recommendation: FIX-AND-REBASELINE.** Numerical padding parity fails in the historical width-24 implementation, and the separate coordinate restoration crosses the locked diagnostic materiality rule. The causes now have explicit matched audit variants; historical selection comparisons need a corrected baseline before simple controls can proceed. This is a Gate A decision, not a submission decision.

## Executed and not executed

| Item | Execution in this batch | Evidence |
|---|---|---|
| Environment recovery and exact package inventory | Executed; usable Python3.11 found, no installation/download | audit/environment_gpu.json; {r}/environment_gpu_recovered_v2.json |
| Five FoB hashes and raw/adapted strict loads; SAM-B strict load | Executed; raw FoB strict failures retained | checkpoint_schema_verification.json; sam_b_strict_load.json |
| Official upstream source identity | Executed read-only; FoB.py and encoder.py match official source after EOL normalization | upstream_source_identity.json and source snapshot |
| Numerical upstream/local parity widths10/24 | Executed; exact width10, width24 fails | parity_width10/24.json and logs |
| Refinement failure isolation | Executed as separately named ring-only and trimmed-refinement arms | parity_width24_validring/trimmed.json |
| 256→257 sensitivity | Executed on eight committed, fixed slices; full ten only | protocol_lock/; coordinate_diagnostic/ |
| CPU tests previously blocked at collection | Executed; {len(pytest)} testcase executions passed, {pytest.test_name.nunique()} distinct names | audit/test_status_v2.csv and four JUnit files |
| Active Step4/5 completeness | Executed; 120 episode identities and five saved Step4 arms | active_completeness.xml |
| Saved-mask independent Dice/3D HD95; disjoint raw/CC decomposition | Executed for all1,200 arm/post/episode rows | metrics_full/; corrected_analysis/ |
| Candidate logging and exact roundtrip | Executed through actual prediction wrapper functions with mock decoders | logging_tests.xml; candidate_records.py |
| SAM-H official weight identity | UNVERIFIED: checkpoint/release bytes unavailable in inspected paths | No weight-identity claim |
| Step5 prefix/random/GT-reference independent mask verification | UNVERIFIED except k0/k10 correspondence; masks absent, including zip | step5_mask_inventory.json |
| Generic legacy GPU allocator/query-label tests | Not executed; uncached bootstrap and scope; no pass inferred | explicit SKIPPED ledger rows |
| Full120-episode new inference; selection methods; MRI; training | Not executed | Batch stops at this report |
| Cold-start and cached-embedding timing | Not measured | Only scenario below has timing evidence |

All artifact paths without a directory prefix in this report are under {r}. SKIPPED ledger entries explicitly mean not executed; they are not successful assertions. The legacy test_fixed_budget_parity repeats one randomly initialized implementation and checks counts/repeatability; it cannot certify upstream numerical parity. Its CPU count test executed separately.

## Environment, authorization and weights

The default primary Python3.12 remains dependency-limited. Recovery used the already installed Python3.11 host behind the Windows Store launcher, invoked directly through the node-host subprocess runtime after exec_command failed before shell startup. No package installation, checkpoint bootstrap download or new architecture was introduced.

{md(pd.DataFrame(package_rows))}

Full installed distributions and Python/platform/CUDA/device/precision are pinned in audit/environment_gpu.json. GPU computations/timing use only NVIDIA GeForce RTX4050 Laptop, CUDA12.8, float32, autocast off, TF32 off, deterministic cuDNN, two torch CPU threads. The guarded unavailable InfoNCE is training-only and raises if invoked; it was not invoked in inference. Redundant ResNet initialization loads are bypassed only during construction; checkpoint state supplies every used parameter.

The supplied Kaggle token passed a read-only HTTP200 authentication probe. It is stored outside the repository with restricted access and omitted from logs. Installed CLI1.7.4.5 has legacy username/key authentication rather than the inspected access_token workflow; the probe used bearer authentication. No remote job or data upload was needed or launched. No T4 timing is inferred from RTX4050 measurements.

All cv0..cv4 SHA256 values match the Step6 records. See checkpoint_schema_verification.json for complete hashes and per-implementation records. **Every original raw FoB strict load fails** because encoder.reduce2.weight is absent. Zero unexpected keys were observed. Encoder.forward never uses reduce2; an explicit unused_reduce2_removed schema adapter deletes that unused registered layer and then loads strictly with zero missing/unexpected keys. This adapter is reported separately from original raw strict loading. At width24 the existing loader also pads head.head.3.weight, head.head.3.bias and masked_attention.conv.weight; their before/after shapes are recorded. It is not a byte-identical checkpoint schema.

SAM-B loads strictly with zero missing/unexpected keys and SHA256 ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912, matching the recorded value. SAM-H architecture evidence does not establish weight identity; that remains UNVERIFIED.

The executed vendored upstream files match official primebo1/FoB_SAM sources after CRLF normalization (FoB blob73b253219e9235c002945b4c6739364b80827a83; encoder blobc37e225d641119f3a91cf20d89f93bde4f3b611e). The main repository's git SHA is not misrepresented as an upstream repository commit.

## Locked tolerance and real numerical parity

Protocol timestamp before numerical results: {lock["timestamp_utc_before_numerical_results"]}. Absolute tolerance1e-5, relative tolerance0; relative differences are descriptive with denominator max(abs(reference),1e-8). Binary/CC masks require exact equality. These tolerances were not relaxed after failures.

The manifest and tolerance were committed in an **isolated local protocol-snapshot Git repository** under protocol_lock before inference; commit metadata is in protocol_lock_commit.json. Main .git is read-only and main HEAD did not change. This is not a retrospective preregistration of Step4/5, or a main-repository method-confirmation freeze.

Same cv0/SAM-B checkpoints, synthetic inputs and seed42; actual upstream and local forwards; same positive/negative points per corresponding implementation, R1/token0, preprocessing, image embedding, threshold0 and CC. The test compares all emitted positive/negative coordinates, postprocessed/low-resolution logits, quality and final masks.

{pd.DataFrame(parity).to_markdown(index=False,floatfmt=".17g")}

Width10 is exact across all tensors and all intermediate hooks. Width24 first diverges at SPR refinement: encoder, masked attention and first-ten heatmaps remain exact. Padded topology makes valid channels0 and9 lose their circular adjacency (links go to invalid23/10 and are masked), and padded tensor operations also change numerical execution.

{pd.DataFrame(fixes).to_markdown(index=False,floatfmt=".17g")}

Ring-only repair reduces but does not eliminate discrepancies and still fails the original tolerance. Running SPR only on the ten valid channels restores exact parity without changing the encoder, candidate heatmaps, checkpoint or preprocessing. This is a named audit-only matched fixed arm in ring_diagnostic.py; production FoB source remains unchanged. These findings isolate padded refinement, rather than a coordinate-order, dtype, image-encoder or preprocessing difference. Synthetic final masks happen to agree even when logits fail; this is not cohort-level equivalence.

## Coordinate restoration sensitivity

Separate variant: x_native=(x_256+0.5)*257/256-0.5, likewise y. Both positive and negative points are restored; legacy behavior remains available. This is the pixel-centre inverse of the resize geometry, with x/y versus row/column and non-square axis scaling unit-tested. No candidates are added or moved by a selection policy.

Eight fixed median organ-positive slices: image_0 and image_7, four organs each, including a known difficult patient intentionally. They are exploratory diagnostics, not fresh holdout patients. Support fold/part/native-slice assignments are locked. Whole-volume normalization, support-mask interpolation, frozen weights, R1 and CC match legacy behavior; query GT is used only for declared slice/metric diagnostics, never model prompting.

{md(pd.DataFrame(sensitivity))}

One of eight CC comparisons crosses the **predeclared** rule (|Dice delta|≥0.005 or |slice HD95 delta|≥1mm or changed image fraction≥0.005). image_7/RK CC changes +0.863007pp, −2.433565mm and34 mask pixels; image_7/LIVER CC changes −0.479882pp. The result is bidirectional and does not demonstrate a cohort improvement.

Negative shifts average {np.average(negative.mean_shift_px,weights=negative["count"]):.6f}px over80 points; maximum {negative.max_shift_px.max():.6f}px / {negative.max_shift_mm.max():.6f}mm. Relative to each same legacy predicted boundary, all80 negative points change their interpolated boundary-distance measurement; none crosses the boundary side or the1mm membership threshold in these eight cases. Distances are to a predicted boundary, not proof of background correctness. Full per-slice raw/CC distance changes and out-of-bounds counters are saved.

All eight upstream10/local10 raw and CC masks agree. Legacy24 versus width10 changes one raw pixel in each of two cases and one CC pixel in one case; this bounds only this small diagnostic, not full-cohort effects. Historical coordinates/IDs are unavailable, so these are current matched diagnostic records, not reconstructed historical trajectories.

These are **2D slice** surface distances with verified in-plane spacing. They must not be presented as the3D volume HD95 endpoint. The legacy grid mismatch is a material sensitivity confound for small Step5 control effects; its contribution to random>prefix or the+4.49pp GT-guided gain remains UNVERIFIED until paired rebaselining. Eight slices cannot establish negligible cohort effects.

## Corrected saved-mask metrics and estimands

No new inference in this section. Existing240 archives cover120 episodes, five arms, raw/CC. Saved GT matches reconstructed target-active slabs. Physical spacing is verified and z indices are contiguous; volumes are the resampled benchmark grid/slab, not an original-acquisition whole-scan evaluation.

Production metrics execute the actual compute_metrics AST from run_step4_baselines.py, using installed MedPy EDT. Independent Dice uses direct counts; independent3D HD95 uses physical surface-voxel centres and a KD-tree nearest-neighbour calculation in both directions. Surface=foreground minus6-neighbour erosion, symmetric HD95=95th percentile of the pooled two directed distance arrays, matching the benchmark operator. This is not maximum of directed percentiles or surface-area-weighted mesh distance.

All1,200 rows execute independently. Maximum absolute source-CSV Dice discrepancy={facts["dice_max_csv_delta"]:.17g}; maximum production HD95 discrepancy={facts["production_hd95_max_csv_delta"]:.17g}mm. Independent finite HD95 agrees with production within1e-6mm by assertion. Empty prediction rows={facts["empty_prediction_rows"]}; independent nonfinite surface rows={facts["nonfinite_independent_surface_rows"]}; full-miss rows={facts["full_miss_rows"]}. No case/image_7/outlier was excluded.

{md(pd.DataFrame(primary))}

Episode weighting is benchmark-compatible. Patient balancing averages repeats within patient-organ, macro-averages all four organs within each patient, then averages25 patients. The120 episodes are30 patient-fold occurrences, not120 independent patients. Both estimands, all per-episode values, median/p90/p95 HD95, physical FP/FN volumes and recall are saved; corrected CSV quantities retain units.

Production empty masks were tested directly: both empty→Dice1/HD0; one empty→Dice0/volume-diagonal HD. Diagonal uses voxel counts times spacing (not index extents). Exceptions, NaN/Inf and values above the diagonal return/clip to the diagonal. These are compatibility fallbacks, not geometrically valid HD for empty sets. Independent one-empty surface HD is infinity and would be retained with an explicit status; none is silently discarded. The old copied300mm test does not certify this production policy. Production metric ASTs remain unchanged by logging instrumentation.

### True disjoint FP partition and CC removals

For each raw or CC mask separately, use3D6-connected prediction components:
1. detached_fp: FP in a prediction component with zero GT overlap;
2. other_annotated_organ_fp: remaining FP overlapping any other annotated organ;
3. residual_fp: all remaining FP.

The ordering is explicit; detached components take precedence over other-organ overlap. Categories are disjoint and union exactly the FP mask, asserted for every1,200 rows. They are operational sets, without verified anatomical or causal mechanisms. The2D8-connected positive-anchored CC filter can retain several components; it does not guarantee one GT-connected3D component.

Raw voxels removed by CC are recorded separately as removed_tp and removed_fp. For all600 episode-arm pairs, rawTP=ccTP+removedTP and rawFP=ccFP+removedFP. Removed voxels are not added to residual CC FP categories.

{md(pd.DataFrame(part))}

Entries are episode means in voxels. Removed columns describe the same raw→CC operation in both post rows, not extra residual errors. Patient-balanced versions and mm3 quantities are in corrected_analysis/decomposition_both_estimands.csv.

### Surface-to-surface versus historical surface-to-foreground

The legacy directed fields sampled the boundary-to-foreground EDT, assigning zero to a predicted boundary inside GT (or the reverse). They were not directed surface distances. Reconstruction matches the historical B0/B3B fields within1e-6, while the true replacement uses nearest surface centres.

{md(pd.DataFrame(direct))}

All values are episode-mean mm. The symmetric primary HD95 was already a true MedPy surface metric; changing the diagnostic directed fields does not mean replacing an incorrect primary HD95 implementation. Clinical labels such as boundary leakage/wrong-organ mechanism/HD95 driver are withdrawn; overlap alone is not a verified explanation.

### Historical numbers and interpretations that change

{md(pd.DataFrame(changes))}

Per-row old/new values are retained in historical_field_replacements.csv; nothing is overwritten. Corrected fields have different operational definitions, not silently reused historical names. All Step4 raw/CC Dice and primary HD95 CSV values remain numerically valid for their **legacy predictions**. The Step5 B0 CC HD95 prose24.52mm must be replaced by25.100679mm (rounded25.10), supported by independent recomputation.

Step5 subset/random/GT-reference masks were never saved in the inspected tree or artifact zip. Thus their Dice/HD CSV values cannot be independently re-evaluated here. The k0/k10 rows exactly correspond to saved Step4 B4B/B3B. Both estimands were rerun for all Step5 CSV endpoints with that limitation:

{md(s5table)}

GT-guided greedy/prefix remain non-exhaustive per-slice raw-Dice references; no exact ceiling or post-CC optimum is inferred. Paired exploratory5,000-resample query-patient intervals/contrasts are in the two paired JSON files. They condition on fixed support selection and checkpoints, omit support/training uncertainty, and are pointwise rather than multiplicity-adjusted confirmation intervals. No nonsignificance/equivalence claim is made.

## Invalidated and sensitivity-required evidence

| Evidence | Status after this batch |
|---|---|
| Claim that padded width24 exactly reproduces upstream fixed10 | INVALIDATED by numerical parity |
| Raw FoB checkpoint strict-load pass | INVALIDATED; actual failures and explicit unused-layer adapter recorded |
| Step5 categories as a disjoint CC FP partition | INVALIDATED; removed TP/FP and overlapping categories corrected |
| Legacy directed h95 fields as surface-to-surface/clinical error drivers | INVALIDATED; operator and causal interpretation corrected |
| B0 H/R3 and B3H/B3B R1 negative-prompt results as unchanged upstream reproduction | SENSITIVITY-REQUIRED: checked-in runners use local FoB default24; exact historical candidate pool was not logged |
| B4H/B4B positive-only arms | No direct negative-SPR-padding dependence; positive-grid restoration SENSITIVITY-REQUIRED |
| Step5 prefix/random/full/GT-guided comparisons and contamination/proximity summaries | SENSITIVITY-REQUIRED for corrected coordinates/pool; missing subset masks independently UNVERIFIED |
| Primary Dice/HD95 of saved legacy masks | VALID as fixed-prediction legacy measurements, not corrected inference |
| SAM-H actual official weight identity | UNVERIFIED until bytes available |
| Random advantage caused by clustering; achievable greedy headroom | UNVERIFIED; no dispersion/response mechanism or deployable selector tested |

The strongest alternative explanation is a combination of privileged query-GT per-slice search and standard budget/order effects, under an altered padded candidate generator and mismatched application grid, followed by a different CC/volume objective. The padding defect has a small observed mask effect in this diagnostic; the coordinate defect is material in one case. **Neither finding proves that the entire selection signal is an artifact**, and the surviving CSV headroom does not establish a deployable contribution.

## Practical cost and minimal rebaseline plan

All timing uses only RTX4050 Laptop; no concurrent GPU job. One warm-up, three synchronized repetitions, identical float32 resolution/batching. Models and volume slices are resident; the measured pipeline includes transfers, FoB generation, coordinate restoration, uint8 conversion, fresh SAM image encoding, R1 decoding and CC. Disk reads, model construction, logging export and volume metrics are excluded and must be added to a complete production timing. No latency saving is inferred from prompt count.

| Scenario | Repetitions seconds | Mean seconds | Median seconds | Sample SD seconds | Peak allocated bytes |
|---|---|---:|---:|---:|---:|
| Resident models, uncached embedding, one full-ten slice | {timing["seconds"]} | {timing["mean_seconds"]:.6f} | {timing["median_seconds"]:.6f} | {timing["std_seconds"]:.6f} | {timing["peak_allocated_bytes"]} |

This is a representative pilot, not a full-dataset/cold-start/cached-embedding latency table. The diagnostic executed32 decoder evaluations plus4 timing evaluations, with one image embedding per diagnostic slice and four fresh timing embeddings. Four parity arms add8 decoder evaluations; no hidden single-pass claim.

Minimum proposed next batch, requiring approval of the concrete compute estimate before launch:
- Freeze the full-ten valid-channel FoB variant (width10, or the exact-parity trimmed-SPR audit fix) and pixel-centre restoration, same cv0..cv4 weights, SAM-B R1, support/positive generation, preprocessing and CC.
- Rebaseline only paired SAM-B full-ten and positive-only on the existing120 episodes; save masks, both coordinate grids, IDs, counts, failures and all decoder calls. Keep legacy masks as the separate legacy arm. No rule search or learned selector.
- At the measured mean, one4,009-slice uncached-embedding arm is {timing["estimated_4009_slice_seconds_at_mean"]:.3f}s ({timing["estimated_4009_slice_seconds_at_mean"]/60:.2f}min; {timing["estimated_4009_slice_seconds_at_mean"]/3600:.3f}GPU-hours). Two separately executed arms conservatively budget {upper/60:.2f}min before loading/I/O/scoring overhead. Shared encoding may reduce this, but that saving is not measured.
- Two raw/CC uint8 masks per arm for4,009x257x257 voxels require approximately {4009*257*257*4/1e9:.3f}GB uncompressed for two arms, plus GT/logs. Compression is not assumed. Cache one embedding at a time; no persistent full-cohort embedding cache. One GPU worker; no simultaneous timing jobs.
- Do not extrapolate RTX4050 throughput to T4x2. If using Kaggle, first measure T4 throughput/VRAM under the same settings, then assign disjoint folds to identical devices and keep timing tables separate.
- Reassess Gate A on the corrected paired baseline. Only a subsequent authorized batch may proceed to simple controls. If Step5 reference/control effects must be reassessed, specify a bounded paired rerun first; no unrestricted full greedy matrix is approved or launched here.

## Preservation and stop point

historical_hash_recheck.json verifies448/450 prior hashes unchanged; the two intended logging-only runner changes have exact original copies in source_before and a tracked diff. Step4/Step5 result CSV hashes still match Step6. Production metric ASTs are unchanged. All historical predictions, checkpoints, manifests and reports remain in place. New numerical results/logs occupy {r}; the original parity driver is also preserved.

No selector, response-score prototype, candidate relocation policy, adaptation/training, second benchmark or manuscript text was implemented. Gate A has a diagnosed failure and an exact-parity minimal repair arm, but corrected cohort inference has deliberately not been launched. This batch ends with the single recommendation **FIX-AND-REBASELINE**.
"""
 path=Path("reports/step7_gate_a.md")
 with path.open("x",encoding="utf-8") as f:f.write(text)
 print(json.dumps({"report":str(path),"Gate_A":"FAIL","recommendation":"FIX-AND-REBASELINE","pytest_executions":len(pytest),"distinct_pytest_names":pytest.test_name.nunique()}),flush=True)
if __name__=="__main__":main()
