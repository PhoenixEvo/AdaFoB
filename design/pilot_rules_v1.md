# Pilot rules v1: approved Route B, no inference in preparation stage

Activation requires an immutable isolated protocol commit containing this document, scoring_v1.py, test_route_b_v1.py, slice_api.py and the manifest, followed by executed CPU tests and explicit current GPU authorization. No parameters may be changed after check outcomes are seen. Artifact hashes are external audit records. The current decision remains CONTINUE-RESEARCH; no manuscript drafting.

## Fixed data and backbone

Use protocol/pilot_manifest_v1.csv: image8 construction, image20 check, image14 support; one support slice per base organ. The lower median of the ascending native zero-based positive z indices resolves an even-count median deterministically. No novel masks, historical oracle files or historical novel-organ utility labels enter these rules. This is an owner-approved amendment, reported separately from Setting I. Prior exposure and possible unlabeled pretraining overlap preclude untouched-data claims.

Use cv0 FoB checkpoint only, width10 valid-channel SPR matched variant, original ten generated negative channels at Np10. No candidate movement or expansion, and no changed positive generation. Record exact frozen inference source and checkpoint hashes from Step8/9. R2 pixel-centre restoration applies to positives and negatives: (x256+.5)*native_width/256-.5, analogously y. Preserve out-of-bounds values and counters, never clip candidate records. SAM-B single output token0 (R1), IoU quality ignored for mask choice; mask logit threshold0. Existing positive-anchored per-slice CC unchanged. SAM-H official weight identity remains UNVERIFIED and SAM-H is excluded.

Keep normalized FoB image preprocessing, uint8 SAM conversion, SAM resize/pad/normalization and postprocessing exactly as analysis/rebaseline_v1/common.py, matched.py and frozen Step9 runner; do not introduce a new standardization. The slice API takes normalized query/support arrays and support mask only; benchmark assembly chooses the prespecified slices outside that API. Query-label independence is slice-level, conditional on these inputs, not whole-volume localization independence.

## Frozen evidence, no fitted coefficients

Use frozen SAM image-encoder outputs CxHxW for the allowed support slice and query slice, before mask decoding; retain both embeddings, one query encoder call per slice. One support encoding per distinct support slice, reusable across the two query patients. Align the binary support mask to the feature grid by nearest pixel-centre sampling in the unpadded square native geometry (all manifested native grids must be square); non-square input blocks this version rather than silently ignoring SAM padding. A feature-grid center maps to native (index+.5)*native_size/feature_size-.5 with nearest index floor(x+.5), bounded only for array indexing. No query GT is used in this resampling.

L2-normalize every feature vector, average normalized support vectors within (i) 3x3-eroded support foreground and (ii) the complement of 3x3-dilated support foreground, then L2-normalize each descriptor. Exclude zero-norm support vectors. Require both nonempty descriptor sets, finite vectors, nonzero descriptor means, nonzero query norms and distinct descriptors. The only numerical guard is float32 machine epsilon. No reliability threshold is fitted. Thin organs without a reliable core fall back. Remaining finite but ambiguous evidence is explicitly uncertain, not verified background.

Evidence e = (cos(query,foreground)-cos(query,background))/2, clipped only to numerical [-1,1]. Bilinearly interpolate the evidence grid to the native mask grid using pixel-centre mapping, nearest-edge extension. This similarity contrast is NOT a calibrated probability. Support anatomical labels are allowed runtime inputs; query anatomical labels are not. A distinct support patient prevents trivial self-mask matching. Implementation: scoring_v1.evidence.

Let M be full10 raw mask, Mi the raw mask after deleting negative i. Ai=Mi AND NOT M (added after deletion); Li=M AND NOT Mi (lost after deletion). Store these separately. The normalized signed sums are ai=sum(Ai*e)/N and li=sum(Li*e)/N, with N the full slice pixel count. Do not threshold e into confirmed foreground/background.

All subsets retain k=2 original negative IDs. This deliberately matches boundary_near2; k2 is not claimed optimal. Rank decreasing retained-point score, ties lower original channel ID, then pass selected coordinates to SAM in ascending original ID order:

- magnitude: |Ai union Li|/N, without a utility interpretation;
- support-at-coordinate: -e(xi,yi), bilinear sampling in restored native xy;
- signed-region: li-ai (negative estimated deletion benefit);
- foreground-protected: same signed ranking but veto retained points when sum(Ai*max(e,0)) exceeds sum(Li*max(e,0)). This hard rule uses no tuned weight. It penalizes a negative whose deletion restores more support-compatible foreground than it loses.

If support evidence is invalid, any candidate is nonfinite/outside the native evidence grid, or fewer than two candidates survive protection, use the FULL ORIGINAL TEN in the affected support-based arm. The magnitude-only arm can remain defined if masks are valid. Record fallback reason and actual count10; this exception is not a matched-k2 observation. Zero signed response ties select IDs0,1. No selection uses a query metric or retrospective utility. No output diversity, weighting search, learned score or parameter tuning.

## Logical requests and caching

Maximum20 requests per slice: full10 once; ten single deletions in ID0..9 order; boundary_near2; magnitude2; coordinate2; signed2; protected2; random2 at seeds42,43,44,45 using NumPy default_rng.choice without replacement. Use existing analysis.controls_v1.core.proximity against the FULL10 raw predicted boundary for near2, including its declared ID tie handling; no GT boundary. Record dispersion in restored native pixel coordinates. All scoring uses the raw response masks, independent of CC.

Cache ordered ID tuples; canonical ascending original IDs make duplicate configurations identical. Reuse masks/logits for duplicates. Report 20 logical requests, actual unique prompt configurations <=20, Python/API invocations, batching policy, encoder calls and all actual counts separately. Batched calls count each configuration. No positive-only or extra context decode is added. No batching modification is necessary for this small sequential pilot.

After selecting, jointly decode every unique requested subset; never sum isolated effects as joint quality. Retrospective deletion benefit = raw Dice(Mi,G)-raw Dice(M,G), positive means deletion helped. Compute only after committed rules and saved predictions. Report where LOO-derived preference and the actually decoded k2 outcomes disagree, without claiming the ten singleton utilities characterize all point interactions.

## Interpretation locked before results

Primary descriptive check-patient comparisons: jointly decoded CC Dice AND recall for each scoring arm against near2, mean of four random k2 masks' metrics, and full10. Also report raw Dice, recall, FP/FN pixel counts and physical area, actual IDs, fallback/count, individual random arms, trajectories and response signs. Optional surface distance is explicitly 2D symmetric slice surface HD95 in verified in-plane spacing; never native-volume 3D HD95. No significance tests, confidence/generalization or learnable-headroom claims from these eight slices.

Stop the selector direction if signed-region evidence adds nothing over coordinate evidence or near2 on the check patient. To propose expansion, protected signed-region must have strictly higher macro mean CC Dice than BOTH coordinate and near2, and mean check recall at least each comparator; disclose all per-organ decreases. Also require higher mean Dice than mean-random and measurable total cost. A result failing these checks is negative or inconclusive; do not tune the rule on these cases. Meeting them is descriptive functionality evidence only, not confirmation. Larger patient-separated development and confirmation plans require separate approval and a newly frozen minimum useful effect. No automatic pivot to mask selection.