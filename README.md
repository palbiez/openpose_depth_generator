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
|-- convert_18_to_25.py
|-- convert_18_to_25_and_normalize.py
|-- copy_missing_depth_inputs.py
|-- dedup_openposer.py
|-- flatten_dataset.py
|-- pose_selection.py
|-- pose_selection2.py
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

2. Convert and normalize keypoints

   Use `convert_18_to_25.py` or `convert_18_to_25_and_normalize.py` to convert 18-keypoint inputs to 25 keypoints. Inputs with more than 25 keypoints are clamped by the conversion logic where supported.

3. Flatten filenames

   Use `flatten_dataset.py` when nested folder paths need to be encoded into flat filenames so SMPLify-X can reliably find matching images and keypoints.

4. Route normal vs. complex poses

   Use `pose_selection.py` to move matching images, keypoints and render folders into:

   ```text
   dataset/poses_normal/
   dataset/poses_complex/
   ```

   Normal poses are intended for VPoser-assisted fitting. Complex poses such as kneeling, all-fours, lying, split-leg and dynamic poses should use a more conservative configuration.

5. Fit SMPL-X

   Use the SMPLify-X configs in the submodule:

   ```text
   smplify-x/cfg_files/fit_smplx_normal.yaml
   smplify-x/cfg_files/fit_smplx_komplex.yaml
   ```

6. Render passes in Blender

   Import generated SMPL-X meshes locally and render depth, lineart and normal passes. Generated meshes and parameter files stay private.

7. Quality control

   Use `dedup_openposer.py` and manual review to quarantine duplicate, invalid or already processed inputs. Complex poses and head/neck deformation need extra review.

## Script Notes

- `copy_missing_depth_inputs.py` copies missing input pairs from a source OpenPose tree into the local dataset.
- `convert_18_to_25.py` converts 18-keypoint JSON files to 25-keypoint JSON files and flattens output names.
- `convert_18_to_25_and_normalize.py` converts 18-keypoint JSON files while preserving the relative folder layout.
- `flatten_dataset.py` copies nested keypoint and image files into flat target folders.
- `pose_selection.py` routes full asset bundles into `poses_normal` or `poses_complex`.
- `pose_selection2.py` routes images based on existing JSON classification.
- `dedup_openposer.py` moves processed or invalid inputs into `dataset/backup/`.

Before running scripts, check the path constants at the top of each file.

## Known Issues

- Fits can produce unstable heads when face keypoints are missing.
- VPoser helps normal upright poses but can over-constrain complex poses.
- Kneeling, all-fours, lying and split-leg poses often need separate fitting settings and stricter review.
- Some earlier local outputs may be invalid because of old keypoint-count or model-path issues.

## License Notes

The code in this repository is separate from the licenses for SMPL, SMPL-X, VPoser, `human_body_prior` and any downloaded pretrained assets. Those assets must be obtained and used under their own licenses.

Rendered outputs are generally safer to publish than original model assets or reconstructable geometry, but review the generated dataset before publication.
