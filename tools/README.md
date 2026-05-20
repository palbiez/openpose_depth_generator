# Helper Tools

Run these from the repository root with `py -3 .\tools\<script>.py ...`.

| Script | Purpose |
| --- | --- |
| `convert_18_to_25_and_normalize.py` | Converts BODY_18 keypoint JSON files into the BODY_25 layout used by this pipeline. |
| `convert_18_to_25.py` | Legacy BODY_18 to BODY_25 conversion without the newer normalization tweaks. |
| `flatten_dataset.py` | Flattens nested OpenPose folders into stable pipeline filenames. |
| `pose_selection2.py` | Routes flattened files into `dataset/poses_normal` and `dataset/poses_complex`. |
| `repair_bone_structure.py` | Audits and redraws bone-structure PNGs from valid OpenPose JSON files. |
| `augment_body25_face_keypoints.py` | Synthesizes 68 face landmarks from BODY_25 head anchors for SMPLify-X. |
| `prepare_head_pose_batches.py` | Creates review batches for weak or missing head/face keypoints. |
| `audit_rendered_outputs.py` | Audits rendered depth/lineart/normal outputs and can quarantine bad renders. |
| `dedup_openposer.py` | Finds and quarantines duplicate routed inputs. |
| `export_openpose_outputs.py` | Copies rendered passes back into the nested OpenPose model tree and creates release backups. |

Most tools write reports under `dataset/`. The dataset folder is intentionally ignored by Git.
