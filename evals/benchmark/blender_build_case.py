"""Blender-side benchmark case builder (runs INSIDE Blender background mode).

Usage:
    blender --background --factory-startup \
        --python evals/benchmark/blender_build_case.py -- <spec_json> <out_dir>

The spec JSON (written by generate_benchmark.py) selects the object builder,
difficulty, seed and render settings. The script produces in <out_dir>:

    input.png       beauty render (Eevee, studio setup)
    mask_raw.png    white-emission override render (object only, black bg)
    depth.png       16-bit normalised camera-depth render (object only)
    normals.png     camera-space normal render (object only, n*0.5+0.5)
    reference.glb   ground-truth 3D model
    camera.json     exact intrinsics/extrinsics + depth/normal encodings

Blender 5.2 notes (verified by probes):
- Eevee engine id is 'BLENDER_EEVEE' (no _NEXT suffix).
- Render-pass pixels are not accessible headless and the scene compositor
  API changed, so depth/normals/mask are material-override renders instead.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector

# ---------------------------------------------------------------------------
# spec loading
# ---------------------------------------------------------------------------

argv = sys.argv[sys.argv.index("--") + 1:]
SPEC_PATH, OUT_DIR = argv[0], argv[1]
with open(SPEC_PATH, "r") as f:
    SPEC = json.load(f)

os.makedirs(OUT_DIR, exist_ok=True)
RNG = random.Random(SPEC["seed"])
RES = int(SPEC.get("resolution", 640))
DIFF = SPEC.get("difficulty", "easy")


# ---------------------------------------------------------------------------
# scene reset + helpers
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mat_principled(name, color, roughness=0.6, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return m


def link_object(name, mesh):
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def smooth(mesh):
    for p in mesh.polygons:
        p.use_smooth = True


def recalc_outside(mesh):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()


def add_bevel(ob, width=0.008, segments=2):
    mod = ob.modifiers.new("EdgeBevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    return mod


def add_cube(name, dims, loc=(0, 0, 0), rot=(0, 0, 0), mat=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    ob.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        ob.data.materials.append(mat)
    if bevel > 0:
        add_bevel(ob, bevel)
    return ob


def add_cylinder(name, radius, depth, loc=(0, 0, 0), rot=(0, 0, 0),
                 mat=None, vertices=48, bevel=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
                                        depth=depth, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    if bevel > 0:
        add_bevel(ob, bevel)
    return ob


def add_cone(name, r1, r2, depth, loc=(0, 0, 0), rot=(0, 0, 0), mat=None):
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=r1, radius2=r2,
                                    depth=depth, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    return ob


def add_uv_sphere(name, radius, loc=(0, 0, 0), mat=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    smooth(ob.data)
    return ob


def build_revolve(name, profile, segments, mat=None):
    """Spin a (r, z) profile (bottom -> top) around the Z axis."""
    bottom_pole = abs(profile[0][0]) < 1e-9
    top_pole = abs(profile[-1][0]) < 1e-9
    core = profile[:]
    if bottom_pole:
        core = core[1:]
    if top_pole:
        core = core[:-1]
    m = len(core)
    verts = []
    if bottom_pole:
        verts.append((0.0, 0.0, profile[0][1]))
    base = len(verts)
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        ca, sa = math.cos(a), math.sin(a)
        for (r, z) in core:
            verts.append((r * ca, r * sa, z))
    top_idx = None
    if top_pole:
        top_idx = len(verts)
        verts.append((0.0, 0.0, profile[-1][1]))
    faces = []
    for i in range(segments):
        ni = (i + 1) % segments
        if bottom_pole:
            faces.append((0, base + ni * m, base + i * m))
        for j in range(m - 1):
            a = base + i * m + j
            b = base + ni * m + j
            faces.append((a, b, b + 1, a + 1))
        if top_pole:
            faces.append((top_idx, base + i * m + m - 1, base + ni * m + m - 1))
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    recalc_outside(mesh)
    ob = link_object(name, mesh)
    if mat:
        mesh.materials.append(mat)
    smooth(mesh)
    return ob


def torus_profile(center_r, center_z, tube_r, steps=20):
    return [(center_r + tube_r * math.cos(2 * math.pi * i / steps),
             center_z + tube_r * math.sin(2 * math.pi * i / steps))
            for i in range(steps)]


def build_prism(name, poly2d, depth, loc=(0, 0, 0), rot=(0, 0, 0), mat=None):
    """Extrude a 2D XY polygon along Z by ``depth`` (centred)."""
    n = len(poly2d)
    hz = depth / 2.0
    verts = [(x, y, -hz) for (x, y) in poly2d] + [(x, y, hz) for (x, y) in poly2d]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    recalc_outside(mesh)
    ob = link_object(name, mesh)
    ob.location = loc
    ob.rotation_euler = rot
    if mat:
        mesh.materials.append(mat)
    return ob


def build_sweep(name, points, radius, mat=None, resolution=3):
    """Sweep a circular section along a polyline path."""
    cu = bpy.data.curves.new(name + "_curve", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = resolution
    cu.resolution_u = 2
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for i, p in enumerate(points):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
    ob = link_object(name, cu)
    if mat:
        cu.materials.append(mat)
    return ob


def radial_copies(ob, count, axis="Z", pivot=(0.0, 0.0, 0.0)):
    """Duplicate ``ob`` ``count`` times rotated around a pivot axis."""
    from mathutils import Matrix
    pivot_v = Vector(pivot)
    out = [ob]
    for k in range(1, count):
        a = 2.0 * math.pi * k / count
        rot_m = Matrix.Rotation(a, 3, axis)
        dup = ob.copy()
        dup.data = ob.data
        dup.name = "%s_%02d" % (ob.name, k + 1)
        dup.location = rot_m @ (ob.location - pivot_v) + pivot_v
        dup.rotation_euler = (rot_m @ ob.rotation_euler.to_matrix()).to_euler()
        bpy.context.scene.collection.objects.link(dup)
        out.append(dup)
    return out


def boolean_difference(target, cutter):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    mod = target.modifiers.new("Cut", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def star_polygon(cx, cy, r_outer, r_inner, points=5, rot=math.pi / 2):
    poly = []
    for i in range(points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        a = rot + math.pi * i / points
        poly.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return poly


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

MATS = {}


def setup_materials():
    reflective = DIFF == "hard"
    MATS["dark_rubber"] = mat_principled("DarkRubber", (0.03, 0.03, 0.032), 0.85)
    MATS["steel"] = mat_principled("Steel", (0.45, 0.47, 0.50),
                                   0.25 if reflective else 0.4,
                                   0.9 if reflective else 0.6)
    MATS["plastic_red"] = mat_principled("PlasticRed", (0.55, 0.08, 0.06), 0.5)
    MATS["plastic_blue"] = mat_principled("PlasticBlue", (0.08, 0.18, 0.45), 0.5)
    MATS["plastic_green"] = mat_principled("PlasticGreen", (0.10, 0.35, 0.15), 0.55)
    MATS["plastic_white"] = mat_principled("PlasticWhite", (0.80, 0.79, 0.75), 0.5)
    MATS["plastic_gray"] = mat_principled("PlasticGray", (0.35, 0.36, 0.38), 0.55)
    MATS["wood"] = mat_principled("Wood", (0.45, 0.30, 0.15), 0.7)
    MATS["glass_green"] = mat_principled("GlassGreen", (0.12, 0.35, 0.20),
                                         0.15 if reflective else 0.35)
    MATS["ceramic"] = mat_principled("Ceramic", (0.75, 0.72, 0.68), 0.3)
    MATS["clutter"] = mat_principled("Clutter", (0.55, 0.55, 0.57), 0.8)


# ---------------------------------------------------------------------------
# object builders — each returns {"parts": [objects], "aim_z": float}
# ---------------------------------------------------------------------------

def build_wheel():
    parts = []
    tyre = build_revolve("Tyre", torus_profile(0.30, 0.0, 0.115, 24), 64,
                         MATS["dark_rubber"])
    tyre.rotation_euler = (math.radians(90), 0, 0)
    tyre.location = (0, 0, 0.415)
    parts.append(tyre)
    # annular rim (box-section ring) so the spokes stay visible through it
    rim_profile = [(0.07, -0.065), (0.20, -0.065), (0.20, 0.065),
                   (0.07, 0.065), (0.07, -0.065)]
    rim = build_revolve("Rim", rim_profile, 64, MATS["steel"])
    rim.rotation_euler = (math.radians(90), 0, 0)
    rim.location = (0, 0, 0.415)
    parts.append(rim)
    hub = add_cylinder("Hub", 0.06, 0.17, loc=(0, 0, 0.415),
                       rot=(math.radians(90), 0, 0), mat=MATS["plastic_gray"])
    parts.append(hub)
    spoke = add_cube("Spoke_01", (0.27, 0.14, 0.05),
                     loc=(0.135, 0, 0.415), mat=MATS["steel"], bevel=0.005)
    parts.extend(radial_copies(spoke, 5, axis="Y", pivot=(0, 0, 0.415)))
    return {"parts": parts, "aim_z": 0.415}


def build_bottle():
    profile = [(0.0, 0.0), (0.13, 0.0), (0.15, 0.03), (0.15, 0.40),
               (0.135, 0.47), (0.07, 0.54), (0.052, 0.57), (0.052, 0.66),
               (0.062, 0.68), (0.062, 0.71), (0.052, 0.72), (0.052, 0.74),
               (0.0, 0.74)]
    body = build_revolve("Body", profile, 64, MATS["glass_green"])
    cap = add_cylinder("Cap", 0.058, 0.05, loc=(0, 0, 0.765),
                       mat=MATS["plastic_red"], bevel=0.004)
    return {"parts": [body, cap], "aim_z": 0.38}


def build_vase():
    profile = [(0.0, 0.0), (0.10, 0.0)]
    steps = 22
    for i in range(1, steps):
        t = i / steps
        z = 0.02 + 0.56 * t
        r = 0.05 + 0.13 * math.sin(math.pi * min(1.0, t * 1.15)) ** 1.2
        if t > 0.8:  # neck pinch
            r *= 1.0 - 0.45 * (t - 0.8) / 0.2
        profile.append((max(0.045, r), z))
    profile.append((0.075, 0.60))
    profile.append((0.0, 0.60))
    body = build_revolve("Body", profile, 64, MATS["ceramic"])
    return {"parts": [body], "aim_z": 0.30}


def build_mug():
    profile = [(0.0, 0.0), (0.125, 0.0), (0.135, 0.015),
               (0.135, 0.28), (0.125, 0.30), (0.0, 0.30)]
    body = build_revolve("Body", profile, 64, MATS["plastic_blue"])
    handle = build_revolve("Handle", torus_profile(0.085, 0.0, 0.018, 16), 48,
                           MATS["plastic_blue"])
    handle.rotation_euler = (math.radians(90), 0, 0)
    handle.location = (0.175, 0, 0.16)
    return {"parts": [body, handle], "aim_z": 0.16}


def build_gear():
    gear = add_cylinder("GearBody", 0.26, 0.08, loc=(0, 0, 0.04),
                        mat=MATS["steel"])
    tooth = add_cube("Tooth_01", (0.075, 0.055, 0.08), loc=(0.285, 0, 0.04),
                     mat=MATS["steel"], bevel=0.004)
    radial_copies(tooth, 12, axis="Z", pivot=(0, 0, 0.04))
    cutter = add_cylinder("_cutter", 0.06, 0.2, loc=(0, 0, 0.04))
    boolean_difference(gear, cutter)
    parts = [gear, tooth] + [o for o in bpy.context.scene.objects
                             if o.name.startswith("Tooth_") and o is not tooth]
    return {"parts": parts, "aim_z": 0.04}


def build_bracket():
    l_profile = [(0, 0), (0.40, 0), (0.40, 0.12), (0.12, 0.12),
                 (0.12, 0.40), (0, 0.40)]
    ob = build_prism("Bracket", l_profile, 0.12, mat=MATS["plastic_gray"])
    ob.rotation_euler = (math.radians(90), 0, 0)
    ob.location = (-0.20, 0, 0.06)
    add_bevel(ob, 0.005)
    # two mounting holes, one per leg
    c1 = add_cylinder("_c1", 0.035, 0.3, loc=(-0.09, 0, 0.31),
                      rot=(math.radians(90), 0, 0))
    boolean_difference(ob, c1)
    c2 = add_cylinder("_c2", 0.035, 0.3, loc=(0.11, 0, 0.06),
                      rot=(math.radians(90), 0, 0))
    boolean_difference(ob, c2)
    return {"parts": [ob], "aim_z": 0.20}


def build_sign():
    plate = add_cube("Plate", (0.70, 0.05, 0.45), loc=(0, 0, 0.225),
                     mat=MATS["plastic_white"], bevel=0.012)
    star = build_prism("StarRelief", star_polygon(0, 0, 0.13, 0.055), 0.03,
                       loc=(0, -0.04, 0.24), rot=(math.radians(90), 0, 0),
                       mat=MATS["plastic_red"])
    ring = build_revolve("RingRelief", torus_profile(0.165, 0.0, 0.012, 16), 64,
                         MATS["plastic_red"])
    ring.rotation_euler = (math.radians(90), 0, 0)
    ring.location = (0, -0.033, 0.24)
    feet = add_cube("Foot", (0.5, 0.12, 0.03), loc=(0, 0.02, 0.015),
                    mat=MATS["plastic_gray"], bevel=0.005)
    return {"parts": [plate, star, ring, feet], "aim_z": 0.225}


def build_box_enclosure():
    body = add_cube("Body", (0.50, 0.35, 0.30), loc=(0, 0, 0.15),
                    mat=MATS["plastic_gray"], bevel=0.02)
    lid = add_cube("Lid", (0.52, 0.37, 0.06), loc=(0, 0, 0.325),
                   mat=MATS["plastic_blue"], bevel=0.015)
    feet = []
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        feet.append(add_cylinder("Foot_%d" % (i + 1), 0.03, 0.03,
                                 loc=(0.20 * sx, 0.12 * sy, -0.005),
                                 mat=MATS["dark_rubber"]))
    return {"parts": [body, lid] + feet, "aim_z": 0.17}


def build_crate():
    parts = [add_cube("Bottom", (0.50, 0.50, 0.04), loc=(0, 0, 0.02),
                      mat=MATS["wood"], bevel=0.004)]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        parts.append(add_cube("Post_%d" % (i + 1), (0.05, 0.05, 0.36),
                              loc=(0.225 * sx, 0.225 * sy, 0.20),
                              mat=MATS["wood"], bevel=0.003))
    n = 0
    for side in range(4):
        for zi, z in enumerate((0.11, 0.21, 0.31)):
            n += 1
            rot = (0, 0, math.radians(90)) if side % 2 else (0, 0, 0)
            x = 0.245 if side == 1 else (-0.245 if side == 3 else 0)
            y = 0.245 if side == 2 else (-0.245 if side == 0 else 0)
            parts.append(add_cube("Slat_%02d" % n, (0.44, 0.025, 0.06),
                                  loc=(x, y, z), rot=rot, mat=MATS["wood"],
                                  bevel=0.002))
    return {"parts": parts, "aim_z": 0.20}


def build_desk_lamp():
    base = add_cylinder("Base", 0.12, 0.035, loc=(0, 0, 0.018),
                        mat=MATS["plastic_gray"], bevel=0.006)
    lower = add_cylinder("LowerArm", 0.016, 0.32, loc=(0.03, 0, 0.19),
                         rot=(0, math.radians(18), 0), mat=MATS["steel"])
    # lower arm top ends near (0.079, 0, 0.342); upper arm leans +52 deg
    upper = add_cylinder("UpperArm", 0.014, 0.30, loc=(0.197, 0, 0.434),
                         rot=(0, math.radians(52), 0), mat=MATS["steel"])
    shade = add_cone("Shade", 0.10, 0.035, 0.13, loc=(0.30, 0, 0.48),
                     rot=(0, math.radians(232), 0), mat=MATS["plastic_green"])
    bulb = add_uv_sphere("Bulb", 0.028, loc=(0.29, 0, 0.44),
                         mat=MATS["plastic_white"])
    return {"parts": [base, lower, upper, shade, bulb], "aim_z": 0.25}


def build_chair():
    parts = [add_cube("Seat", (0.40, 0.40, 0.04), loc=(0, 0, 0.30),
                      mat=MATS["wood"], bevel=0.008)]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        parts.append(add_cylinder("Leg_%d" % (i + 1), 0.02, 0.30,
                                  loc=(0.17 * sx, 0.17 * sy, 0.15),
                                  mat=MATS["wood"]))
    parts.append(add_cube("Backrest", (0.40, 0.035, 0.10), loc=(0, 0.185, 0.62),
                          mat=MATS["wood"], bevel=0.006))
    for i, x in enumerate((-0.12, 0.0, 0.12)):
        parts.append(add_cube("BackSlat_%d" % (i + 1), (0.05, 0.03, 0.30),
                              loc=(x, 0.185, 0.45), mat=MATS["wood"],
                              bevel=0.004))
    return {"parts": parts, "aim_z": 0.32}


def build_table():
    parts = [add_cube("Top", (0.70, 0.50, 0.045), loc=(0, 0, 0.42),
                      mat=MATS["wood"], bevel=0.008)]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        parts.append(add_cylinder("Leg_%d" % (i + 1), 0.028, 0.42,
                                  loc=(0.30 * sx, 0.20 * sy, 0.21),
                                  mat=MATS["plastic_gray"]))
    return {"parts": parts, "aim_z": 0.22}


def build_knob():
    profile = [(0.0, 0.0), (0.105, 0.0), (0.105, 0.025), (0.05, 0.045),
               (0.045, 0.09), (0.07, 0.13), (0.095, 0.17), (0.075, 0.21),
               (0.03, 0.225), (0.0, 0.225)]
    body = build_revolve("Body", profile, 64, MATS["plastic_red"])
    ring = add_cylinder("BaseRing", 0.11, 0.012, loc=(0, 0, 0.006),
                        mat=MATS["steel"])
    return {"parts": [body, ring], "aim_z": 0.11}


def build_pipe_elbow():
    pts = [(-0.40, 0, 0.30), (-0.30, 0, 0.30), (-0.27, 0, 0.30)]
    for i in range(1, 17):
        t = i / 16.0
        a = math.pi - (math.pi / 2.0) * t
        pts.append((0.25 * math.cos(a) - 0.02, 0, 0.30 + 0.25 * math.sin(a)))
    pts.extend([(0.0, 0, 0.62), (0.0, 0, 0.70)])
    pipe = build_sweep("Pipe", pts, 0.045, MATS["steel"])
    fl_a = add_cylinder("Flange_A", 0.075, 0.04, loc=(-0.36, 0, 0.30),
                        rot=(0, math.radians(90), 0), mat=MATS["steel"])
    fl_b = add_cylinder("Flange_B", 0.075, 0.04, loc=(0.0, 0, 0.64),
                        mat=MATS["steel"])
    return {"parts": [pipe, fl_a, fl_b], "aim_z": 0.35}


BUILDERS = {
    "wheel": build_wheel,
    "bottle": build_bottle,
    "vase": build_vase,
    "mug": build_mug,
    "gear": build_gear,
    "bracket": build_bracket,
    "sign": build_sign,
    "box_enclosure": build_box_enclosure,
    "crate": build_crate,
    "desk_lamp": build_desk_lamp,
    "chair": build_chair,
    "table": build_table,
    "knob": build_knob,
    "pipe_elbow": build_pipe_elbow,
}


# ---------------------------------------------------------------------------
# studio, camera, lights
# ---------------------------------------------------------------------------

DIFF_PRESETS = {
    "easy": dict(lens=50.0, dist=1.9, az=45.0, el=20.0, key_size=1.0,
                 key_energy=250.0, world_strength=0.30, clutter=0),
    "medium": dict(lens=55.0, dist=1.8, az=32.0, el=27.0, key_size=2.5,
                   key_energy=220.0, world_strength=0.28, clutter=1),
    "hard": dict(lens=38.0, dist=1.5, az=58.0, el=34.0, key_size=1.2,
                 key_energy=180.0, world_strength=0.22, clutter=3),
}


def setup_studio(preset, aim_z):
    scn = bpy.context.scene
    # ground
    ground = add_cube("Ground", (30.0, 30.0, 0.1), loc=(0, 0, -0.0501),
                      mat=mat_principled("GroundMat", (0.62, 0.62, 0.63), 0.9))
    # world
    world = bpy.data.worlds.new("StudioWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.75, 0.76, 0.78, 1.0)
    bg.inputs[1].default_value = preset["world_strength"]
    scn.world = world
    # key light
    key = bpy.data.lights.new("KeyLight", "AREA")
    key.energy = preset["key_energy"] * RNG.uniform(0.95, 1.05)
    key.size = preset["key_size"]
    key_ob = bpy.data.objects.new("KeyLight", key)
    key_ob.location = (1.6, -1.8, 2.4)
    scn.collection.objects.link(key_ob)
    point_at(key_ob, (0, 0, aim_z))
    # fill light
    fill = bpy.data.lights.new("FillLight", "AREA")
    fill.energy = preset["key_energy"] * 0.35
    fill.size = 2.0
    fill_ob = bpy.data.objects.new("FillLight", fill)
    fill_ob.location = (-1.8, -1.0, 1.2)
    scn.collection.objects.link(fill_ob)
    point_at(fill_ob, (0, 0, aim_z))
    # clutter slabs behind the object (visible in beauty only)
    clutter = []
    for i in range(preset["clutter"]):
        w = RNG.uniform(0.25, 0.45)
        h = RNG.uniform(0.25, 0.55)
        d = RNG.uniform(0.05, 0.15)
        x = RNG.uniform(-0.9, 0.9)
        y = RNG.uniform(1.3, 1.8)
        clutter.append(add_cube("Clutter_%d" % i, (w, d, h),
                                loc=(x, y, h / 2.0),
                                rot=(0, 0, RNG.uniform(-0.3, 0.3)),
                                mat=MATS["clutter"]))
    return ground, clutter


def point_at(ob, target):
    direction = Vector(target) - ob.location
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(preset, target, dist):
    scn = bpy.context.scene
    az = math.radians(preset["az"] + RNG.uniform(-3.0, 3.0))
    el = math.radians(preset["el"] + RNG.uniform(-2.0, 2.0))
    loc = Vector((dist * math.cos(el) * math.sin(az),
                  -dist * math.cos(el) * math.cos(az),
                  target.z + dist * math.sin(el)))
    cam = bpy.data.cameras.new("Camera")
    cam.lens = preset["lens"]
    cam.sensor_width = 36.0
    cam_ob = bpy.data.objects.new("Camera", cam)
    cam_ob.location = loc
    scn.collection.objects.link(cam_ob)
    point_at(cam_ob, target)
    scn.camera = cam_ob
    return cam_ob


def object_frame(parts):
    """World-space centre + bounding radius of the benchmark parts."""
    bpy.context.view_layer.update()  # refresh matrix_world for new objects
    pts = []
    for o in parts:
        if o.type not in ("MESH", "CURVE"):
            continue
        for corner in o.bound_box:
            pts.append(o.matrix_world @ Vector(corner))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts),
                 min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts),
                 max(p.z for p in pts)))
    center = (lo + hi) / 2.0
    radius = max((p - center).length for p in pts)
    return center, radius


# ---------------------------------------------------------------------------
# override materials for mask / depth / normals
# ---------------------------------------------------------------------------

def emission_override(name, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def depth_override(dmin, dmax):
    m = bpy.data.materials.new("DepthOverride")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    # Eevee 5.2 returns camera-space Z positive in front of the camera;
    # ABSOLUTE is robust to either sign convention.
    ab = nt.nodes.new("ShaderNodeMath")
    ab.operation = "ABSOLUTE"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = dmin
    mr.inputs["From Max"].default_value = dmax
    mr.clamp = True
    nt.links.new(tc.outputs["Camera"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ab.inputs[0])
    nt.links.new(ab.outputs[0], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def normals_override():
    m = bpy.data.materials.new("NormalsOverride")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    vt = nt.nodes.new("ShaderNodeVectorTransform")
    vt.vector_type = "NORMAL"
    vt.convert_from = "WORLD"
    vt.convert_to = "CAMERA"
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    nt.links.new(geo.outputs["Normal"], vt.inputs["Vector"])
    nt.links.new(vt.outputs["Vector"], mul.inputs[0])
    nt.links.new(mul.outputs[0], add.inputs[0])
    nt.links.new(add.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def base_render_settings():
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE"
    scn.render.resolution_x = RES
    scn.render.resolution_y = RES
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    scn.render.film_transparent = False


def render_to(path, color_depth="8"):
    scn = bpy.context.scene
    scn.render.image_settings.color_depth = color_depth
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)


def set_black_world():
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    old = (tuple(bg.inputs[0].default_value), bg.inputs[1].default_value)
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 0.0
    return old


def main():
    reset_scene()
    setup_materials()
    preset = DIFF_PRESETS[DIFF]
    built = BUILDERS[SPEC["builder"]]()
    # frame the object so its bounding radius fills ~half the frame
    target, radius = object_frame(built["parts"])
    tan_half_fov = 18.0 / preset["lens"]           # sensor 36mm, square frame
    dist = min(5.0, max(0.6, radius / (0.65 * tan_half_fov)))
    aim_z = target.z
    ground, clutter = setup_studio(preset, aim_z)
    cam_ob = setup_camera(preset, target, dist)
    base_render_settings()
    scn = bpy.context.scene
    view_layer = scn.view_layers[0]

    part_names = sorted(o.name for o in built["parts"])
    # tag benchmark parts so downstream tooling can identify them
    for o in built["parts"]:
        o["benchmark_part"] = True

    # reference 3D model
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT_DIR, "reference.glb"),
                              export_format="GLB", export_apply=True)

    # 1) beauty
    render_to(os.path.join(OUT_DIR, "input.png"))

    # hide ground + clutter for GT passes
    hidden = [ground] + clutter
    for o in hidden:
        o.hide_render = True
    set_black_world()

    # 2) mask (white emission on black)
    view_layer.material_override = emission_override("MaskOverride", (1, 1, 1))
    render_to(os.path.join(OUT_DIR, "mask_raw.png"))

    # 3) depth (normalised camera distance, 16-bit)
    cam_dist = (cam_ob.location - target).length
    dmin = cam_dist - radius - 0.15
    dmax = cam_dist + radius + 0.15
    view_layer.material_override = depth_override(dmin, dmax)
    render_to(os.path.join(OUT_DIR, "depth.png"), color_depth="16")

    # 4) normals (camera space, *0.5+0.5)
    view_layer.material_override = normals_override()
    render_to(os.path.join(OUT_DIR, "normals.png"))

    view_layer.material_override = None

    # camera ground truth
    cam = cam_ob.data
    focal_px = cam.lens / cam.sensor_width * RES
    camera_gt = {
        "resolution": [RES, RES],
        "engine": "BLENDER_EEVEE",
        "projection": "perspective",
        "lens_mm": cam.lens,
        "sensor_width_mm": cam.sensor_width,
        "focal_length_px": focal_px,
        "principal_point_px": [RES / 2.0, RES / 2.0],
        "camera_location": list(cam_ob.location),
        "camera_rotation_quaternion_wxyz": list(cam_ob.rotation_quaternion),
        "camera_matrix_world": [list(r) for r in cam_ob.matrix_world],
        "look_at_target": [target.x, target.y, target.z],
        "camera_distance": cam_dist,
        "frame_radius": radius,
        "depth_min": dmin,
        "depth_max": dmax,
        "depth_encoding": ("16-bit PNG; d = depth_min + v/65535*(depth_max-depth_min); "
                           "0 = background"),
        "normals_encoding": ("8-bit PNG; n_cam = v/255*2-1 (camera space); "
                             "0,0,0 = background"),
        "part_object_names": part_names,
        "difficulty": DIFF,
        "seed": SPEC["seed"],
    }
    with open(os.path.join(OUT_DIR, "camera.json"), "w") as f:
        json.dump(camera_gt, f, indent=2)

    print("BUILD_OK parts=%d" % len(built["parts"]))


main()
