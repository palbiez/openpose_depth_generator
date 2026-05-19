from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import traceback
from math import atan, radians
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMPLX_MODEL_PATH = PROJECT_ROOT / "dataset" / "smplx" / "SMPLX_FEMALE.npz"
PASS_NAMES = {"depth", "lineart", "normal"}
SMPLX_HEAD_LANDMARK_IDS = {
    "nose": 9120,
    "reye": 9929,
    "leye": 9448,
    "rear": 616,
    "lear": 6,
}


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_resolution(raw: str) -> tuple[int, int]:
    value = raw.lower().replace(" ", "")
    for separator in ("x", ",", ":"):
        if separator in value:
            width_raw, height_raw = value.split(separator, 1)
            width = int(width_raw)
            height = int(height_raw)
            break
    else:
        width = int(value)
        height = width

    if width <= 0 or height <= 0:
        raise ValueError("Resolution values must be positive")

    return width, height


def parse_vector(raw: str) -> Vector:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Expected XYZ vector formatted as x,y,z")
    return Vector((float(parts[0]), float(parts[1]), float(parts[2])))


def parse_xyz_degrees(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Expected XYZ rotation formatted as x,y,z degrees")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resolution", default="512x768")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--passes", default="depth,lineart,normal")
    parser.add_argument("--view", choices=["front", "back", "left", "right", "top"], default="front")
    parser.add_argument(
        "--camera-mode",
        choices=["orthographic", "smplifyx"],
        default="orthographic",
        help="Use a generic orthographic view or the saved SMPLify-X perspective camera.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=5000.0,
        help="Perspective focal length in pixels for --camera-mode smplifyx.",
    )
    parser.add_argument(
        "--smplifyx-auto-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale/center SMPLify-X perspective renders so the mesh stays in frame.",
    )
    parser.add_argument(
        "--smplifyx-frame-padding",
        type=float,
        default=1.04,
        help="Frame padding multiplier for --smplifyx-auto-frame.",
    )
    parser.add_argument(
        "--obj-axis-mode",
        choices=["smplx-y-up", "raw"],
        default="smplx-y-up",
        help="OBJ import axes. smplx-y-up maps SMPLify-X Y-up OBJ files into Blender Z-up.",
    )
    parser.add_argument(
        "--mesh-rotation",
        default="0,0,0",
        help="Additional XYZ rotation in degrees after OBJ import, e.g. 90,0,0.",
    )
    parser.add_argument("--padding", type=float, default=1.45)
    parser.add_argument("--line-thickness", type=float, default=1.0)
    parser.add_argument("--normal-space", choices=["camera", "world"], default="camera")
    parser.add_argument("--head-mode", choices=["original", "proxy", "replace"], default="original")
    parser.add_argument("--head-scale", type=float, default=1.0)
    parser.add_argument("--head-offset", default="0,0,0")
    parser.add_argument("--smplx-model-path", default=str(DEFAULT_SMPLX_MODEL_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--auto-upright", action="store_true")
    args = parser.parse_args(blender_args())
    args.passes = [name.strip().lower() for name in args.passes.split(",") if name.strip()]
    invalid = sorted(set(args.passes) - PASS_NAMES)
    if invalid:
        raise ValueError(f"Unsupported pass: {', '.join(invalid)}")

    if args.width is None or args.height is None:
        args.width, args.height = parse_resolution(args.resolution)
    args.head_offset_vec = parse_vector(args.head_offset)
    args.mesh_rotation_degrees = parse_xyz_degrees(args.mesh_rotation)

    return args


def set_color_management() -> None:
    scene = bpy.context.scene
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
    except Exception:
        pass


def set_engine() -> None:
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
        try:
            scene.render.engine = engine
            return
        except Exception:
            continue


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.images,
    ):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def import_obj(path: Path, axis_mode: str) -> list[bpy.types.Object]:
    clear_scene()

    if axis_mode == "smplx-y-up":
        forward_axis = "NEGATIVE_Z"
        up_axis = "Y"
        legacy_forward_axis = "-Z"
        legacy_up_axis = "Y"
    elif axis_mode == "raw":
        forward_axis = "Y"
        up_axis = "Z"
        legacy_forward_axis = "Y"
        legacy_up_axis = "Z"
    else:
        raise ValueError(f"Unsupported OBJ axis mode: {axis_mode}")

    if hasattr(bpy.ops.wm, "obj_import"):
        try:
            bpy.ops.wm.obj_import(
                filepath=str(path),
                forward_axis=forward_axis,
                up_axis=up_axis,
            )
        except TypeError:
            bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(
            filepath=str(path),
            axis_forward=legacy_forward_axis,
            axis_up=legacy_up_axis,
        )

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")

    bpy.context.view_layer.update()
    return meshes


def rotate_meshes(objects: list[bpy.types.Object], degrees: tuple[float, float, float]) -> None:
    if all(abs(value) < 1e-6 for value in degrees):
        return

    rotation = Euler(tuple(radians(value) for value in degrees), "XYZ")
    for obj in objects:
        obj.rotation_euler.rotate(rotation)

    bpy.context.view_layer.update()


def auto_upright_meshes(objects: list[bpy.types.Object], view: str) -> bool:
    if view not in {"front", "back"}:
        return False

    landmark_estimate = estimate_head_from_landmarks(objects)
    if landmark_estimate is None:
        return False

    head_center, _, _ = landmark_estimate
    min_v, max_v, _ = mesh_bounds(objects)
    body_center_z = (min_v.z + max_v.z) * 0.5

    if head_center.z >= body_center_z:
        return False

    rotate_meshes(objects, (180.0, 0.0, 0.0))
    return True


def mesh_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector, list[Vector]]:
    corners: list[Vector] = []
    for obj in objects:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)

    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_v, max_v, corners


def look_at(obj: bpy.types.Object, target: Vector, up_axis: str = "Y") -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", up_axis).to_euler()


def view_position(view: str, min_v: Vector, max_v: Vector, distance: float) -> tuple[Vector, str, float, float]:
    center = (min_v + max_v) * 0.5

    if view == "front":
        return Vector((center.x, min_v.y - distance, center.z)), "Y", min_v.y, max_v.y
    if view == "back":
        return Vector((center.x, max_v.y + distance, center.z)), "Y", max_v.y, min_v.y
    if view == "left":
        return Vector((min_v.x - distance, center.y, center.z)), "X", min_v.x, max_v.x
    if view == "right":
        return Vector((max_v.x + distance, center.y, center.z)), "X", max_v.x, min_v.x
    if view == "top":
        return Vector((center.x, center.y, max_v.z + distance)), "Z", max_v.z, min_v.z

    raise ValueError(f"Unsupported view: {view}")


def setup_camera(
    objects: list[bpy.types.Object],
    view: str,
    padding: float,
    resolution_x: int,
    resolution_y: int,
) -> tuple[bpy.types.Object, str, float, float]:
    min_v, max_v, corners = mesh_bounds(objects)
    center = (min_v + max_v) * 0.5
    diagonal = max((max_v - min_v).length, 0.1)
    distance = diagonal * 2.0

    camera_data = bpy.data.cameras.new("RenderCamera")
    camera = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    camera.location, depth_axis, near_coord, far_coord = view_position(view, min_v, max_v, distance)
    look_at(camera, center, up_axis="Y" if view == "top" else "Z")

    bpy.context.view_layer.update()

    inv_camera = camera.matrix_world.inverted()
    local_corners = [inv_camera @ corner for corner in corners]
    width = max(v.x for v in local_corners) - min(v.x for v in local_corners)
    height = max(v.y for v in local_corners) - min(v.y for v in local_corners)
    aspect = resolution_x / max(resolution_y, 1)

    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(height, width / aspect, 0.1) * padding
    camera_data.clip_start = 0.001
    camera_data.clip_end = diagonal * 8.0

    return camera, depth_axis, near_coord, far_coord


def flatten_numeric(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    out: list[float] = []
    for item in value:  # type: ignore[union-attr]
        out.extend(flatten_numeric(item))
    return out


def load_smplifyx_camera(pkl_path: Path) -> tuple[Matrix, Vector]:
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")

    if "camera_rotation" not in data or "camera_translation" not in data:
        raise RuntimeError(f"Missing camera_rotation/camera_translation in {pkl_path}")

    rotation_values = flatten_numeric(data["camera_rotation"])
    translation_values = flatten_numeric(data["camera_translation"])
    if len(rotation_values) < 9 or len(translation_values) < 3:
        raise RuntimeError(f"Invalid SMPLify-X camera arrays in {pkl_path}")
    if not all(math.isfinite(value) for value in rotation_values[:9] + translation_values[:3]):
        raise RuntimeError(f"Non-finite SMPLify-X camera values in {pkl_path}")

    rotation = Matrix(
        (
            rotation_values[0:3],
            rotation_values[3:6],
            rotation_values[6:9],
        )
    )
    translation = Vector(translation_values[0:3])
    return rotation, translation


def projected_pixel_bounds(
    vertices: list[Vector],
    resolution_x: int,
    resolution_y: int,
    focal_length: float,
) -> tuple[float, float, float, float]:
    pixels: list[tuple[float, float]] = []
    for vert in vertices:
        if not all(math.isfinite(value) for value in (vert.x, vert.y, vert.z)):
            raise RuntimeError("Cannot project SMPLify-X mesh: non-finite vertex coordinates")
        depth = -vert.z
        if depth <= 1e-6:
            continue
        pixels.append(
            (
                focal_length * vert.x / depth + resolution_x * 0.5,
                resolution_y * 0.5 - focal_length * vert.y / depth,
            )
        )

    if not pixels:
        raise RuntimeError("Cannot project SMPLify-X mesh: all vertices are behind the camera")

    xs = [pixel[0] for pixel in pixels]
    ys = [pixel[1] for pixel in pixels]
    return min(xs), max(xs), min(ys), max(ys)


def translate_meshes(objects: list[bpy.types.Object], offset: Vector) -> None:
    if offset.length < 1e-8:
        return
    transform = Matrix.Translation(offset)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def auto_upright_smplifyx_meshes(objects: list[bpy.types.Object]) -> bool:
    landmark_estimate = estimate_head_from_landmarks(objects)
    if landmark_estimate is None:
        return False

    head_center, _, _ = landmark_estimate
    min_v, max_v, _ = mesh_bounds(objects)
    body_center = (min_v + max_v) * 0.5

    head_vector = head_center - body_center
    # Preserve the source bone-structure image orientation. Only fix the
    # obvious upside-down case where the head is below the body center and the
    # body/head axis is already near vertical. Do not rotate lying or diagonal
    # poses into an artificial standing pose.
    if head_vector.y >= 0:
        return False

    verticality = abs(head_vector.y) / max(head_vector.length, 1e-6)
    if verticality < 0.85:
        return False

    rotation = Matrix.Rotation(radians(180.0), 4, "Z")
    transform = Matrix.Translation(body_center) @ rotation @ Matrix.Translation(-body_center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world

    bpy.context.view_layer.update()
    return True


def auto_frame_smplifyx_meshes(
    objects: list[bpy.types.Object],
    resolution_x: int,
    resolution_y: int,
    focal_length: float,
    padding: float,
) -> tuple[float, dict[str, object]]:
    padding = max(padding, 1.0)
    vertices = world_vertices(objects)
    before = projected_pixel_bounds(vertices, resolution_x, resolution_y, focal_length)
    min_x, max_x, min_y, max_y = before
    box_width = max(max_x - min_x, 1.0)
    box_height = max(max_y - min_y, 1.0)
    scale = min(
        1.0,
        resolution_x / (box_width * padding),
        resolution_y / (box_height * padding),
    )
    effective_focal = max(focal_length * scale, 1.0)

    for _ in range(3):
        vertices = world_vertices(objects)
        min_x, max_x, min_y, max_y = projected_pixel_bounds(
            vertices,
            resolution_x,
            resolution_y,
            effective_focal,
        )
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        depths = sorted(-vert.z for vert in vertices if -vert.z > 1e-6)
        if not depths:
            break
        depth = depths[len(depths) // 2]
        dx = (resolution_x * 0.5 - center_x) * depth / effective_focal
        dy = (center_y - resolution_y * 0.5) * depth / effective_focal
        offset = Vector((dx, dy, 0.0))
        translate_meshes(objects, offset)
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            break

    after = projected_pixel_bounds(world_vertices(objects), resolution_x, resolution_y, effective_focal)
    return effective_focal, {
        "auto_frame": True,
        "frame_padding": padding,
        "input_focal_length": focal_length,
        "effective_focal_length": effective_focal,
        "auto_frame_scale": scale,
        "projected_bounds_before": list(before),
        "projected_bounds_after": list(after),
    }


def setup_smplifyx_camera(
    objects: list[bpy.types.Object],
    item: dict[str, object],
    resolution_x: int,
    resolution_y: int,
    focal_length: float,
    auto_frame: bool,
    frame_padding: float,
    auto_upright: bool,
) -> tuple[bpy.types.Object, str, float, float, dict[str, object]]:
    raw_pkl_path = item.get("pkl_path")
    if not raw_pkl_path:
        raise RuntimeError(f"Missing pkl_path for {item.get('id', item.get('obj_path'))}")

    pkl_path = Path(str(raw_pkl_path))
    if not pkl_path.exists():
        raise RuntimeError(f"SMPLify-X camera PKL not found: {pkl_path}")

    rotation, translation = load_smplifyx_camera(pkl_path)

    # SMPLify-X exports OBJ vertices after a 180 degree X rotation. Its camera
    # projects the pre-export model in image coordinates. Keep that vertical
    # image convention and only flip depth into Blender's camera -Z direction.
    export_flip = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    camera_depth_flip = Matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))
    linear = camera_depth_flip @ rotation @ export_flip
    offset = camera_depth_flip @ translation
    transform = Matrix(
        (
            (linear[0][0], linear[0][1], linear[0][2], offset.x),
            (linear[1][0], linear[1][1], linear[1][2], offset.y),
            (linear[2][0], linear[2][1], linear[2][2], offset.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world

    bpy.context.view_layer.update()

    auto_upright_applied = auto_upright_smplifyx_meshes(objects) if auto_upright else False

    camera_info: dict[str, object] = {
        "auto_frame": False,
        "auto_upright": auto_upright_applied,
        "input_focal_length": focal_length,
        "effective_focal_length": focal_length,
    }
    effective_focal = focal_length
    if auto_frame:
        effective_focal, frame_info = auto_frame_smplifyx_meshes(
            objects,
            resolution_x,
            resolution_y,
            focal_length,
            frame_padding,
        )
        camera_info.update(frame_info)

    verts = world_vertices(objects)
    if not verts:
        raise RuntimeError("Cannot set SMPLify-X camera without mesh vertices")

    z_values = [vert.z for vert in verts]
    depth_distances = [-value for value in z_values if -value > 0]
    if not depth_distances:
        raise RuntimeError("SMPLify-X camera transform placed mesh behind the Blender camera")

    near_coord = max(z_values)
    far_coord = min(z_values)
    near_distance = max(min(depth_distances) * 0.5, 0.001)
    far_distance = max(max(depth_distances) * 1.5, near_distance + 1.0)

    camera_data = bpy.data.cameras.new("SMPLifyXCamera")
    camera = bpy.data.objects.new("SMPLifyXCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    camera.location = (0.0, 0.0, 0.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle_x = 2.0 * atan(resolution_x / (2.0 * max(effective_focal, 1.0)))
    camera_data.shift_x = 0.0
    camera_data.shift_y = 0.0
    camera_data.clip_start = near_distance
    camera_data.clip_end = far_distance

    bpy.context.view_layer.update()
    return (
        camera,
        "Z",
        near_coord,
        far_coord,
        {
            "pkl_path": str(pkl_path),
            **camera_info,
            "camera_translation": [float(value) for value in translation],
            "camera_rotation": [[float(rotation[row][col]) for col in range(3)] for row in range(3)],
            "clip_start": near_distance,
            "clip_end": far_distance,
        },
    )


def world_vertices(objects: list[bpy.types.Object]) -> list[Vector]:
    verts: list[Vector] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        verts.extend(obj.matrix_world @ vert.co for vert in obj.data.vertices)
    for vert in verts:
        if not all(math.isfinite(value) for value in (vert.x, vert.y, vert.z)):
            raise RuntimeError("Mesh contains non-finite vertex coordinates")
    return verts


def average_vector(values: list[Vector]) -> Vector:
    if not values:
        return Vector((0.0, 0.0, 0.0))
    out = Vector((0.0, 0.0, 0.0))
    for value in values:
        out += value
    return out / len(values)


def estimate_head_from_landmarks(
    objects: list[bpy.types.Object],
) -> tuple[Vector, float, bpy.types.Object] | None:
    landmark_ids = list(SMPLX_HEAD_LANDMARK_IDS.values())
    min_vertex_count = max(landmark_ids) + 1
    candidates = [
        obj for obj in objects
        if obj.type == "MESH" and len(obj.data.vertices) >= min_vertex_count
    ]
    if not candidates:
        return None

    obj = max(candidates, key=lambda item: len(item.data.vertices))
    coords = {
        name: obj.matrix_world @ obj.data.vertices[index].co
        for name, index in SMPLX_HEAD_LANDMARK_IDS.items()
    }
    center = average_vector(list(coords.values()))

    verts = world_vertices(objects)
    min_y = min(v.y for v in verts)
    max_y = max(v.y for v in verts)
    body_height = max(max_y - min_y, 0.1)

    ear_width = (coords["lear"] - coords["rear"]).length
    eye_width = (coords["leye"] - coords["reye"]).length
    nose_to_eye = (coords["nose"] - average_vector([coords["leye"], coords["reye"]])).length
    radius = max(ear_width * 0.62, eye_width * 1.25, nose_to_eye * 1.25, body_height * 0.055)
    radius = min(radius, body_height * 0.085)
    return center, radius, obj


def estimate_head_from_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float, bpy.types.Object]:
    verts = world_vertices(objects)
    if not verts:
        raise RuntimeError("Cannot estimate head proxy without mesh vertices")

    min_v = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_v = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    height = max(max_v.y - min_v.y, 0.1)

    top_threshold = max_v.y - height * 0.18
    top_points = [v for v in verts if v.y >= top_threshold]
    if len(top_points) < 30:
        top_points = verts

    center = average_vector(top_points)
    center.y = max_v.y - height * 0.085
    radius = height * 0.085
    target_obj = max([obj for obj in objects if obj.type == "MESH"], key=lambda item: len(item.data.vertices))
    return center, radius, target_obj


def estimate_head(objects: list[bpy.types.Object]) -> tuple[Vector, float, bpy.types.Object]:
    return estimate_head_from_landmarks(objects) or estimate_head_from_bounds(objects)


def template_head_vertex_ids(model_path: Path) -> set[int]:
    try:
        import numpy as np

        data = np.load(model_path, allow_pickle=True)
        vertices = data["v_template"]
        return set(np.where(vertices[:, 1] > 0.15)[0].astype(int).tolist())
    except Exception:
        return set()


def delete_original_head_region(
    obj: bpy.types.Object,
    center: Vector,
    radius: float,
    model_path: Path,
) -> None:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    head_ids = template_head_vertex_ids(model_path)
    if head_ids:
        verts_to_delete = [vert for vert in bm.verts if vert.index in head_ids]
    else:
        delete_radius = radius * 1.55
        verts_to_delete = [
            vert for vert in bm.verts
            if (obj.matrix_world @ vert.co - center).length <= delete_radius
        ]

    if verts_to_delete:
        bmesh.ops.delete(bm, geom=verts_to_delete, context="VERTS")
        bm.to_mesh(mesh)
        mesh.update()

    bm.free()


def add_head_proxy(
    objects: list[bpy.types.Object],
    scale_multiplier: float,
    offset: Vector,
    replace_original: bool,
    model_path: Path,
) -> bpy.types.Object:
    center, radius, target_obj = estimate_head(objects)
    center += offset

    if replace_original:
        delete_original_head_region(target_obj, center, radius, model_path)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=center)
    proxy = bpy.context.object
    proxy.name = "HeadProxy"
    radius = radius * scale_multiplier
    proxy.scale = (radius * 0.82, radius * 1.08, radius * 0.92)
    bpy.ops.object.shade_smooth()
    bpy.context.view_layer.update()
    return proxy


def set_world_color(color: tuple[float, float, float]) -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = color
    try:
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
            background.inputs["Strength"].default_value = 1.0
    except Exception:
        pass


def assign_material(objects: list[bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def make_emission_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0

    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def make_depth_material(axis: str, near_coord: float, far_coord: float) -> bpy.types.Material:
    material = bpy.data.materials.new("DepthMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    map_range = nodes.new("ShaderNodeMapRange")

    map_range.inputs["From Min"].default_value = near_coord
    map_range.inputs["From Max"].default_value = far_coord
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0
    try:
        map_range.clamp = True
    except Exception:
        pass

    axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs[axis_index], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def make_normal_material(space: str) -> bpy.types.Material:
    material = bpy.data.materials.new("NormalMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    multiply = nodes.new("ShaderNodeVectorMath")
    add = nodes.new("ShaderNodeVectorMath")

    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)

    normal_output = geometry.outputs["Normal"]
    if space == "camera":
        transform = nodes.new("ShaderNodeVectorTransform")
        transform.vector_type = "NORMAL"
        transform.convert_from = "WORLD"
        transform.convert_to = "CAMERA"
        links.new(normal_output, transform.inputs["Vector"])
        normal_output = transform.outputs["Vector"]

    links.new(normal_output, multiply.inputs[0])
    links.new(multiply.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def make_lineart_material() -> bpy.types.Material:
    material = bpy.data.materials.new("LineartBlack")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def configure_render(output_path: Path, width: int, height: int, color_mode: str, color_depth: str) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_path)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = color_depth
    scene.render.use_freestyle = False
    try:
        bpy.context.view_layer.use_freestyle = False
    except Exception:
        pass


def configure_freestyle(thickness: float) -> None:
    scene = bpy.context.scene
    scene.render.use_freestyle = True

    view_layer = bpy.context.view_layer
    try:
        view_layer.use_freestyle = True
        settings = view_layer.freestyle_settings
        if not settings.linesets:
            settings.linesets.new("LineSet")
        line_set = settings.linesets[0]
        for attr, value in (
            ("select_silhouette", True),
            ("select_border", False),
            ("select_crease", False),
            ("select_edge_mark", False),
            ("select_material_boundary", False),
        ):
            if hasattr(line_set, attr):
                setattr(line_set, attr, value)
        line_set.linestyle.color = (0.82, 0.82, 0.82)
        line_set.linestyle.thickness = thickness
    except Exception:
        pass


def render_still(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


def render_item(item: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    obj_path = Path(str(item["obj_path"]))
    outputs = {key: Path(value) for key, value in dict(item["outputs"]).items()}
    rendered: dict[str, str] = {}

    import_axis_mode = "raw" if args.camera_mode == "smplifyx" else args.obj_axis_mode
    meshes = import_obj(obj_path, import_axis_mode)
    auto_upright_applied = False
    camera_info: dict[str, object] = {}

    if args.camera_mode == "orthographic":
        rotate_meshes(meshes, args.mesh_rotation_degrees)
        auto_upright_applied = auto_upright_meshes(meshes, args.view) if args.auto_upright else False

    effective_head_mode = "original" if args.camera_mode == "smplifyx" else args.head_mode
    if effective_head_mode in {"proxy", "replace"}:
        meshes.append(
            add_head_proxy(
                meshes,
                args.head_scale,
                args.head_offset_vec,
                replace_original=effective_head_mode == "replace",
                model_path=Path(args.smplx_model_path),
            )
        )

    if args.camera_mode == "smplifyx":
        camera, depth_axis, near_coord, far_coord, camera_info = setup_smplifyx_camera(
            meshes,
            item,
            args.width,
            args.height,
            args.focal_length,
            args.smplifyx_auto_frame,
            args.smplifyx_frame_padding,
            args.auto_upright,
        )
    else:
        camera, depth_axis, near_coord, far_coord = setup_camera(
            meshes,
            args.view,
            args.padding,
            args.width,
            args.height,
        )

    if "depth" in args.passes and (args.force or not outputs["depth"].exists()):
        assign_material(meshes, make_depth_material(depth_axis, near_coord, far_coord))
        set_world_color((0.0, 0.0, 0.0))
        configure_render(outputs["depth"], args.width, args.height, "BW", "16")
        render_still(outputs["depth"])
        rendered["depth"] = str(outputs["depth"])

    if "normal" in args.passes and (args.force or not outputs["normal"].exists()):
        assign_material(meshes, make_normal_material(args.normal_space))
        set_world_color((0.0, 0.0, 0.0))
        configure_render(outputs["normal"], args.width, args.height, "RGB", "16")
        render_still(outputs["normal"])
        rendered["normal"] = str(outputs["normal"])

    if "lineart" in args.passes and (args.force or not outputs["lineart"].exists()):
        assign_material(meshes, make_lineart_material())
        set_world_color((0.0, 0.0, 0.0))
        configure_render(outputs["lineart"], args.width, args.height, "RGB", "8")
        configure_freestyle(args.line_thickness)
        render_still(outputs["lineart"])
        rendered["lineart"] = str(outputs["lineart"])

    return {
        "id": item["id"],
        "obj_path": str(obj_path),
        "status": "ok",
        "camera": {
            "mode": args.camera_mode,
            "type": camera.data.type,
            "view": args.view,
            "location": list(camera.location),
            "ortho_scale": camera.data.ortho_scale if camera.data.type == "ORTHO" else None,
            "angle_x": camera.data.angle_x if camera.data.type == "PERSP" else None,
            "focal_length": camera_info.get("effective_focal_length", args.focal_length)
            if args.camera_mode == "smplifyx"
            else None,
            "depth_axis": depth_axis,
            "near_coord": near_coord,
            "far_coord": far_coord,
            "obj_axis_mode": import_axis_mode,
            "mesh_rotation": list(args.mesh_rotation_degrees) if args.camera_mode == "orthographic" else [0.0, 0.0, 0.0],
            "head_mode": effective_head_mode,
            "auto_upright": auto_upright_applied,
            **camera_info,
        },
        "rendered": rendered,
    }


def main() -> int:
    args = parse_args()
    set_color_management()
    set_engine()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    results: list[dict[str, object]] = []

    print(f"Rendering {len(items)} item(s)")
    print(f"Passes: {', '.join(args.passes)}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Camera mode: {args.camera_mode}")

    for index, item in enumerate(items, start=1):
        item_id = item.get("id", f"item_{index}")
        print(f"[{index}/{len(items)}] {item_id}")
        try:
            results.append(render_item(item, args))
        except Exception as exc:
            traceback.print_exc()
            results.append(
                {
                    "id": item_id,
                    "obj_path": item.get("obj_path"),
                    "status": "error",
                    "error": str(exc),
                }
            )

    results_path = manifest_path.with_suffix(".results.json")
    results_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    errors = [item for item in results if item.get("status") != "ok"]
    print(f"Results: {results_path}")
    print(f"Errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
