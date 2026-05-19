# openpose_depth_generator

Windows pipeline for generating reproducible SMPL-X fits from OpenPose keypoints and bone-structure images, then rendering depth, lineart and normal passes in Blender.

The repository contains pipeline code and project configuration. It deliberately does not contain SMPL-X models, VPoser checkpoints, pretrained weights, generated SMPL-X meshes or reconstructable pose parameter artifacts.

## Current Scope

- Normalize OpenPose JSON files to the BODY_25-style 25-keypoint layout expected by the fitting pipeline.
- Flatten nested pose folders into stable filenames for batch processing.
- Route poses into `normal` and `complex` batches.
- Fit SMPL-X via the project `smplify-x` fork.
- Keep rendered outputs publishable while keeping model assets and reconstructable outputs private.

The pipeline is currently optimized for a Windows-only workflow. Existing helper scripts use absolute Windows paths and should be adjusted before reuse on another machine.

## Repository Layout

```text
.
|-- README.md
|-- .gitignore
|-- .gitmodules
|-- audit_rendered_outputs.py
|-- augment_body25_face_keypoints.py
|-- convert_18_to_25.py
|-- convert_18_to_25_and_normalize.py
|-- copy_missing_depth_inputs.py
|-- dedup_openposer.py
|-- flatten_dataset.py
|-- pose_selection.py
|-- pose_selection2.py
|-- prepare_head_pose_batches.py
|-- repair_bone_structure.py
|-- run_full_pipeline.py
|-- render_blender_batch.py
|-- blender/
|   `-- render_smplx_passes.py
|-- openpose_pipeline_codex_context.json
|-- openpose_pipeline_license_notes.json
`-- smplify-x/                 # submodule: https://github.com/palbiez/smplify-x.git
```

Local-only folders such as `dataset/`, `human_body_prior/`, model files, checkpoints and generated mesh outputs are ignored by Git.

## Git Setup

Clone with the SMPLify-X fork:

```powershell
git clone --recurse-submodules https://github.com/palbiez/openpose_depth_generator.git
cd openpose_depth_generator
git submodule update --init --recursive
```

If `smplify-x` changes, commit and push inside the submodule first:

```powershell
cd smplify-x
git status
git add <files>
git commit -m "Describe smplify-x change"
git push origin master
cd ..
git add smplify-x
git commit -m "Update smplify-x submodule"
git push origin main
```

Root pipeline changes go to:

```text
https://github.com/palbiez/openpose_depth_generator.git
```

SMPLify-X changes go to:

```text
https://github.com/palbiez/smplify-x.git
```

## Private Assets

Expected local/private data layout:

```text
dataset/
|-- images/
|-- keypoints/
|-- keypoints-18/
|-- normal/
|-- lineart/
|-- smplx/
|-- poses_normal/
|-- poses_complex/
`-- vposer/
```

Do not commit or redistribute:

- `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`, `SMPLX_NEUTRAL.npz`
- SMPL `.pkl` model files
- VPoser checkpoints such as `.ckpt`, `.pt`, `.pth`
- pretrained `human_body_prior` weights
- generated `.obj`, `.fbx`, `.glb`, `.blend`, `.ply`, `.stl` files
- pose, betas, expression or parameter files that can reconstruct body shape or motion

Rendered depth maps, lineart renders, normal maps, masks, OpenPose JSONs, bone-structure PNGs and custom scripts are the intended publishable outputs.

## Pipeline

1. Collect source inputs

   Put OpenPose JSON files and matching bone-structure PNG files under the local `dataset/` tree.

2. Audit and redraw bone structures

   Before fitting, check whether the JSON files contain real BODY_18/BODY_25 keypoints. Some legacy files contain raw sampled lineart points in `pose_keypoints_2d`; those can have 80, 94, 120 or similar point counts and cannot be safely converted into OpenPose skeletons.

   Single-file check with diagnostic point plot:

   ```powershell
   py -3 .\repair_bone_structure.py --json "C:\EasyDiffusion\stable-diffusion\stable-diffusion-webui\models\openpose\sitting\F\nsfw\sitting\sitting_176_openpose.json" --write-diagnostics
   ```

   Batch check for routed pipeline inputs:

   ```powershell
   py -3 .\repair_bone_structure.py --input-root .\dataset\poses_normal\keypoints --input-root .\dataset\poses_complex\keypoints --output-root .\dataset\bone_structure_qc\redrawn --report .\dataset\bone_structure_qc\report.jsonl --write-diagnostics --force
   ```

   Valid 18-point JSONs are converted to BODY_25 for drawing. Valid 25-point JSONs are redrawn directly. Implausible BODY_25 geometry, such as many full-canvas crossing limbs, is reported as invalid and not written as a replacement image unless `--allow-implausible` is used for manual diagnostics.

3. Convert and normalize keypoints

   Use `convert_18_to_25.py` or `convert_18_to_25_and_normalize.py` to convert 18-keypoint inputs to 25 keypoints. Inputs with more than 25 keypoints are clamped by the conversion logic where supported.

4. Flatten filenames

   Use `flatten_dataset.py` when nested folder paths need to be encoded into flat filenames so SMPLify-X can reliably find matching images and keypoints.

5. Route normal vs. complex poses

   Use `pose_selection.py` to move matching images, keypoints and render folders into:

   ```text
   dataset/poses_normal/
   dataset/poses_complex/
   ```

   Normal poses are intended for VPoser-assisted fitting. Complex poses such as kneeling, all-fours, lying, split-leg and dynamic poses should use a more conservative configuration.

6. Add face keypoints from BODY_25 head anchors

   Many source OpenPose JSON files contain only BODY_18/BODY_25 head anchors (`Nose`, `REye`, `LEye`, `REar`, `LEar`) and leave `face_keypoints_2d` empty. SMPLify-X can use face landmarks, but only if `face_keypoints_2d` exists and `use_face: True` is enabled in the fitting config.

   Populate synthetic 68-point `face_keypoints_2d` arrays from the BODY_25 head anchors:

   ```powershell
   py -3 .\augment_body25_face_keypoints.py --apply
   ```

   The script backs up changed keypoint JSON files before writing:

   ```text
   dataset/backup/keypoints_before_face_augment/<timestamp>/
   ```

   Dry-run without writing:

   ```powershell
   py -3 .\augment_body25_face_keypoints.py
   ```

   After augmentation, the current SMPLify-X configs use `use_face: True`. Because this pipeline fits face landmarks without hand landmarks, the local SMPLify-X fork also patches the SMPL-X OpenPose mapping so face landmark indices start at `66` when `use_hands: False`.

7. Fit SMPL-X

   Use the SMPLify-X configs in the submodule:

   ```text
   smplify-x/cfg_files/fit_smplx_normal.yaml
   smplify-x/cfg_files/fit_smplx_komplex.yaml
   ```

   For the full resumable pipeline, prefer `run_full_pipeline.py` instead of starting SMPLify-X manually. It runs one image/keypoint pair at a time, logs failures and continues with the next item.

   Dry-run:

   ```powershell
   py -3 .\run_full_pipeline.py --dry-run
   ```

   Small end-to-end test:

   ```powershell
   py -3 .\run_full_pipeline.py --category complex --limit 1 --fit-maxiters 1 --passes depth lineart --force-fit --force-render
   ```

   Full run:

   ```powershell
   py -3 .\run_full_pipeline.py
   ```

   Full forced rebuild, including cleanup of old meshes, PKL results, OBJ files and rendered PNGs:

   ```powershell
   py -3 .\run_full_pipeline.py --force-fit --force-render --backup-cleaned --show-subprocess-output
   ```

   `--force-fit` alone recomputes meshes/results but does not replace already existing render PNGs. Use `--force-render` whenever depth/lineart/normal outputs must be regenerated.

   Resume after interruption or failure by running the same command again. Existing meshes are skipped, and existing render PNGs are skipped:

   ```powershell
   py -3 .\run_full_pipeline.py
   ```

   Useful recovery options:

   ```powershell
   py -3 .\run_full_pipeline.py --category normal
   py -3 .\run_full_pipeline.py --category complex
   py -3 .\run_full_pipeline.py --start-after some_pose_id
   py -3 .\run_full_pipeline.py --force-fit
   py -3 .\run_full_pipeline.py --force-render
   py -3 .\run_full_pipeline.py --skip-fit
   py -3 .\run_full_pipeline.py --skip-render
   py -3 .\run_full_pipeline.py --skip-fit --force-render --resolution 768x512 --padding 1.45
   py -3 .\run_full_pipeline.py --skip-fit --force-render --obj-axis-mode raw
   py -3 .\run_full_pipeline.py --skip-fit --force-render --mesh-rotation 90,0,0
   py -3 .\run_full_pipeline.py --pose-id some_pose_id --force-fit --force-render --backup-cleaned
   ```

   Logs are written to:

   ```text
   dataset/pipeline_logs/<timestamp>/
   ```

   Each item gets separate `smplifyx.log` and `blender.log` files. The run also writes `events.jsonl`, `summary.json` and `missing_pairs.json` when image/keypoint pairs are incomplete.

8. Render passes in Blender

   Import generated SMPL-X meshes locally and render depth, lineart and normal passes. Generated meshes and parameter files stay private.

   The batch renderer scans these mesh roots by default:

   ```text
   dataset/poses_normal/smplx/meshes/
   dataset/poses_complex/smplx/meshes/
   dataset/smplx/meshes/
   ```

   Run a dry-run first:

   ```powershell
   py -3 .\render_blender_batch.py --dry-run --limit 5
   ```

   Render all pending active OBJ files:

   ```powershell
   py -3 .\render_blender_batch.py
   ```

   Outputs are written to:

   ```text
   dataset/rendered/depth/
   dataset/rendered/lineart/
   dataset/rendered/normal/
   dataset/rendered/_manifests/
   ```

   Useful options:

   ```powershell
   py -3 .\render_blender_batch.py --limit 10 --force
   py -3 .\render_blender_batch.py --passes depth normal
   py -3 .\render_blender_batch.py --resolution 512x768
   py -3 .\render_blender_batch.py --camera-mode smplifyx
   py -3 .\render_blender_batch.py --camera-mode orthographic
   py -3 .\render_blender_batch.py --smplifyx-auto-frame
   py -3 .\render_blender_batch.py --resolution 768x512 --padding 1.45
   py -3 .\render_blender_batch.py --include-backup
   py -3 .\render_blender_batch.py --obj-axis-mode smplx-y-up
   py -3 .\render_blender_batch.py --obj-axis-mode raw
   py -3 .\render_blender_batch.py --mesh-rotation 90,0,0
   py -3 .\render_blender_batch.py --head-mode replace --head-scale 0.85
   py -3 .\render_blender_batch.py --head-mode replace --head-offset 0,0.03,-0.02
   py -3 .\render_blender_batch.py --blender-exe "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
   ```

   The default pipeline render mode is `--camera-mode smplifyx`. It reads the matching `results/<pose_id>/000.pkl` and uses the saved SMPLify-X camera translation/rotation instead of a generic Blender front view. `--smplifyx-auto-frame` keeps the rendered body inside the image frame while preserving the fitted camera direction.

   `--obj-axis-mode smplx-y-up` is the default for old orthographic SMPLify-X OBJ renders. In `--camera-mode smplifyx`, the Blender script imports OBJ files as `raw` internally so the saved camera convention can be reconstructed. Use `--camera-mode orthographic` only for diagnostic renders where a generic view is desired.

   `--head-mode replace` is a legacy diagnostic option for poses without usable head data. In `--camera-mode smplifyx`, the renderer forces the original SMPL-X head to avoid destructive proxy-head replacement.

9. Quality control

   Use `dedup_openposer.py` and manual review to quarantine duplicate, invalid or already processed inputs. Complex poses and head/neck deformation need extra review.

   Audit rendered outputs:

   ```powershell
   py -3 .\audit_rendered_outputs.py --required-passes depth,lineart
   ```

   Strict audit that marks large global-orientation fits and blank lineart as reject:

   ```powershell
   py -3 .\audit_rendered_outputs.py --required-passes depth,lineart --reject-large-orientation --orientation-review-threshold 2.8 --reject-blank-lineart
   ```

   Move rejected renders to quarantine:

   ```powershell
   py -3 .\audit_rendered_outputs.py --required-passes depth,lineart --reject-large-orientation --orientation-review-threshold 2.8 --reject-blank-lineart --apply
   ```

   Create a separate report/batch for poses that still have missing or weak head anchors:

   ```powershell
   py -3 .\prepare_head_pose_batches.py
   ```

## Script Notes

- `copy_missing_depth_inputs.py` copies missing input pairs from a source OpenPose tree into the local dataset.
- `convert_18_to_25.py` converts 18-keypoint JSON files to 25-keypoint JSON files and flattens output names.
- `convert_18_to_25_and_normalize.py` converts 18-keypoint JSON files while preserving the relative folder layout.
- `flatten_dataset.py` copies nested keypoint and image files into flat target folders.
- `pose_selection.py` routes full asset bundles into `poses_normal` or `poses_complex`.
- `pose_selection2.py` routes images based on existing JSON classification.
- `dedup_openposer.py` moves processed or invalid inputs into `dataset/backup/`.
- `repair_bone_structure.py` audits OpenPose JSON files, redraws valid BODY_18/BODY_25 bone-structure PNGs and flags raw sampled point payloads.
- `augment_body25_face_keypoints.py` writes synthetic 68-point `face_keypoints_2d` arrays from BODY_25 nose/eye/ear anchors and backs up changed JSON files.
- `prepare_head_pose_batches.py` reports or copies poses with missing/weak head anchors into separate batches.
- `audit_rendered_outputs.py` audits generated depth/lineart/normal outputs and can quarantine empty or suspicious renders.
- `run_full_pipeline.py` runs SMPLify-X and Blender per item with resume behavior and error logs.
- `render_blender_batch.py` discovers pending OBJ files, writes a render manifest and launches Blender.
- `blender/render_smplx_passes.py` runs inside Blender and renders depth, lineart and normal PNGs.

Before running scripts, check the path constants at the top of each file.

## Known Issues

- Fits can produce unstable heads when `face_keypoints_2d` is missing or empty. Run `augment_body25_face_keypoints.py --apply` before fitting with the current `use_face: True` configs.
- Synthetic face landmarks are approximate. They stabilize head/face fitting, but they are not a replacement for a real face detector.
- VPoser helps normal upright poses but can over-constrain complex poses.
- Kneeling, all-fours, lying and split-leg poses often need separate fitting settings and stricter review.
- Large `global_orient` values in the fitted PKL often indicate that SMPLify-X used global body rotation to satisfy ambiguous 2D keypoints. Use `audit_rendered_outputs.py` to flag these cases.
- Some earlier local outputs may be invalid because of old keypoint-count or model-path issues.
- JSON files with arbitrary point counts such as 90 or 94 in `pose_keypoints_2d` are usually raw sampled image points, not OpenPose BODY_25. They should be quarantined or regenerated from a real pose estimator instead of being clamped to 25 points.

## License Notes

The code in this repository is separate from the licenses for SMPL, SMPL-X, VPoser, `human_body_prior` and any downloaded pretrained assets. Those assets must be obtained and used under their own licenses.

Rendered outputs are generally safer to publish than original model assets or reconstructable geometry, but review the generated dataset before publication.
