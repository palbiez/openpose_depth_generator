# Dataset Layout

This directory is intentionally mostly empty in Git.

Versioned files here document the expected local layout and render settings. The actual source images, OpenPose JSON files, SMPL-X model assets, fitted meshes, PKL files and rendered PNG outputs are ignored because they are either large, private, licensed separately or reconstructable.

Expected local folders:

```text
dataset/
|-- images/
|-- keypoints/
|-- keypoints-18/
|-- poses_normal/
|-- poses_complex/
|-- rendered/
|-- smplx/
`-- vposer/
```

The main camera/render defaults are documented in `render_settings.json` and `config/pipeline.example.json`.
