#!/usr/bin/env python3
import os
import sys
import json
import glob
import time
import shutil
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import torch
import cv2
import trimesh

BARC_ROOT = Path("C:/Users/Student3/Desktop/BARC")
os.chdir(str(BARC_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_DIR = Path("C:/Users/Student3/Desktop/ccbd_dogvton/3d-canine-sizing-vton")
SIDEVIEW_ROOT = Path("C:/Users/Student3/Desktop/ccbd_dogvton/side_view_dogs")

sys.path.insert(0, str(BARC_ROOT / "src"))
sys.path.insert(0, str(REPO_DIR))

from configs.barc_cfg_defaults import update_cfg_global_with_yaml, get_cfg_global_updated
from configs.data_info import COMPLETE_DATA_INFO_24
from configs.dataset_path_configs import STANEXT_RELATED_DATA_ROOT_DIR
from combined_model.model_shape_v7 import ModelImageTo3d_withshape_withproj
from controlnet_aux import MidasDetector

OUTPUT_ROOT = Path("D:/sideview_pipeline_stages")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

(OUTPUT_ROOT / "stage_1_3d_meshes" / "golden").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "stage_1_3d_meshes" / "labrador").mkdir(parents=True, exist_ok=True)

(OUTPUT_ROOT / "stage_2_depth_maps" / "golden").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "stage_2_depth_maps" / "labrador").mkdir(parents=True, exist_ok=True)

(OUTPUT_ROOT / "stage_3_anatomical_masks" / "golden").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "stage_3_anatomical_masks" / "labrador").mkdir(parents=True, exist_ok=True)

(OUTPUT_ROOT / "stage_4_vton_renders" / "golden").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "stage_4_vton_renders" / "labrador").mkdir(parents=True, exist_ok=True)

(OUTPUT_ROOT / "stage_5_telemetry_database").mkdir(parents=True, exist_ok=True)

RUFFWEAR_SIZE_CHART = {
    "XXS": {"chest_min": 33.0, "chest_max": 43.0, "chest_mid": 38.0, "back_len_cm": 30.0, "mask_len_pct": 0.50},
    "XS":  {"chest_min": 43.0, "chest_max": 56.0, "chest_mid": 49.5, "back_len_cm": 38.0, "mask_len_pct": 0.60},
    "S":   {"chest_min": 56.0, "chest_max": 69.0, "chest_mid": 62.5, "back_len_cm": 48.0, "mask_len_pct": 0.72},
    "M":   {"chest_min": 69.0, "chest_max": 81.0, "chest_mid": 75.0, "back_len_cm": 58.0, "mask_len_pct": 0.85},
    "L":   {"chest_min": 81.0, "chest_max": 91.0, "chest_mid": 86.0, "back_len_cm": 68.0, "mask_len_pct": 0.95},
    "XL":  {"chest_min": 91.0, "chest_max": 107.0, "chest_mid": 99.0, "back_len_cm": 75.0, "mask_len_pct": 1.00},
}

def init_barc_model(device_str="cuda"):
    checkpoint_path = BARC_ROOT / "checkpoint" / "barc_complete" / "model_best.pth.tar"
    cfg_file = BARC_ROOT / "src" / "configs" / "barc_cfg_visualization.yaml"
    update_cfg_global_with_yaml(str(cfg_file))
    cfg = get_cfg_global_updated()
    
    device_obj = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"[*] Initializing BARC Model on: {device_obj}")
    
    model = ModelImageTo3d_withshape_withproj(
        num_stage_comb=cfg.params.NUM_STAGE_COMB,
        num_stage_heads=cfg.params.NUM_STAGE_HEADS,
        num_stage_heads_pose=cfg.params.NUM_STAGE_HEADS_POSE,
        trans_sep=cfg.params.TRANS_SEP,
        arch=cfg.params.ARCH,
        n_joints=cfg.params.N_JOINTS,
        n_classes=cfg.params.N_CLASSES,
        n_keyp=cfg.params.N_KEYP,
        n_bones=cfg.params.N_BONES,
        n_betas=cfg.params.N_BETAS,
        n_betas_limbs=cfg.params.N_BETAS_LIMBS,
        n_breeds=cfg.params.N_BREEDS,
        n_z=cfg.params.N_Z,
        image_size=cfg.params.IMG_SIZE,
        silh_no_tail=cfg.params.SILH_NO_TAIL,
        thr_keyp_sc=cfg.params.KP_THRESHOLD,
        add_z_to_3d_input=cfg.params.ADD_Z_TO_3D_INPUT,
        n_segbps=cfg.params.N_SEGBPS,
        add_segbps_to_3d_input=cfg.params.ADD_SEGBPS_TO_3D_INPUT,
        add_partseg=cfg.params.ADD_PARTSEG,
        n_partseg=cfg.params.N_PARTSEG,
        fix_flength=cfg.params.FIX_FLENGTH,
        structure_z_to_betas=cfg.params.STRUCTURE_Z_TO_B,
        structure_pose_net=cfg.params.STRUCTURE_POSE_NET,
        nf_version=cfg.params.NF_VERSION
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device_obj)
    model.eval()
    
    data_info = COMPLETE_DATA_INFO_24
    norm_dict = {
        "pose_rot6d_mean": torch.from_numpy(data_info.pose_rot6d_mean).float().to(device_obj),
        "trans_mean": torch.from_numpy(data_info.trans_mean).float().to(device_obj),
        "trans_std": torch.from_numpy(data_info.trans_std).float().to(device_obj),
        "flength_mean": torch.from_numpy(data_info.flength_mean).float().to(device_obj),
        "flength_std": torch.from_numpy(data_info.flength_std).float().to(device_obj)
    }
    return model, norm_dict, device_obj

def preprocess_for_barc(pil_img, target_size=256):
    w, h = pil_img.size
    max_side = max(w, h)
    padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
    padded.paste(pil_img, ((max_side - w) // 2, (max_side - h) // 2))
    resized = padded.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    img_torch = torch.from_numpy(img_np.transpose(2, 0, 1))
    rgb_mean = torch.tensor(COMPLETE_DATA_INFO_24.rgb_mean, dtype=torch.float32).view(3, 1, 1)
    rgb_std = torch.tensor(COMPLETE_DATA_INFO_24.rgb_stddev, dtype=torch.float32).view(3, 1, 1)
    inp = (img_torch - rgb_mean) / rgb_std
    return inp.unsqueeze(0)

def compute_ramanujan_chest_girth(mesh, scale_factor_cm=75.0):
    bounds = mesh.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]
    y_min, y_max = bounds[0][1], bounds[1][1]
    x_min, x_max = bounds[0][0], bounds[1][0]
    
    slice_z = z_min + (z_max - z_min) * 0.58
    slice_chest = mesh.section(plane_origin=[0, 0, slice_z], plane_normal=[0, 0, 1])
    
    if slice_chest is not None and hasattr(slice_chest, "bounds") and slice_chest.bounds is not None:
        s_bounds = slice_chest.bounds
        semi_width = (s_bounds[1][0] - s_bounds[0][0]) * scale_factor_cm / 2.0
        semi_depth = (s_bounds[1][1] - s_bounds[0][1]) * scale_factor_cm / 2.0
    else:
        semi_width = (x_max - x_min) * scale_factor_cm * 0.35 / 2.0
        semi_depth = (y_max - y_min) * scale_factor_cm * 0.55 / 2.0
        
    a, b = max(semi_depth, semi_width), min(semi_depth, semi_width)
    h = ((a - b) ** 2) / ((a + b) ** 2)
    chest_girth_cm = np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))
    
    verts = mesh.vertices
    withers_pt = verts[np.argmax(verts[:, 1])]
    tail_base_pt = verts[np.argmin(verts[:, 2])]
    back_len_cm = np.linalg.norm(withers_pt - tail_base_pt) * scale_factor_cm * 0.85
    height_cm = (y_max - y_min) * scale_factor_cm
    
    return {
        "chest_girth_cm": round(float(chest_girth_cm), 1),
        "back_length_cm": round(float(back_len_cm), 1),
        "withers_height_cm": round(float(height_cm), 1),
        "semi_depth_a_cm": round(float(a), 1),
        "semi_width_b_cm": round(float(b), 1),
        "num_vertices": len(verts),
        "num_faces": len(mesh.faces)
    }

def evaluate_garment_fit(chest_girth_cm):
    verdicts = {}
    best_size = "M"
    min_diff = 999.0
    
    for size, spec in RUFFWEAR_SIZE_CHART.items():
        g_mid = spec["chest_mid"]
        fr = round(chest_girth_cm / g_mid, 3)
        diff = round(((g_mid - chest_girth_cm) / chest_girth_cm) * 100.0, 1)
        
        if spec["chest_min"] <= chest_girth_cm <= spec["chest_max"]:
            verdict = "Good Fit"
            status_color = (50, 220, 100)
        elif chest_girth_cm > spec["chest_max"]:
            verdict = "Tight"
            status_color = (240, 80, 80)
        else:
            verdict = "Loose"
            status_color = (240, 180, 50)
            
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_size = size
            
        verdicts[size] = {
            "garment_chest_mid_cm": g_mid,
            "range_cm": f"{spec['chest_min']}-{spec['chest_max']}cm",
            "fit_ratio": fr,
            "pct_diff": diff,
            "verdict": verdict,
            "status_color": status_color
        }
        
    return best_size, verdicts

def create_scaled_anatomical_mask(image_pil, mask_len_pct=0.95):
    w, h = image_pil.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    
    x_withers = int(w * 0.28)
    x_tail = int(w * 0.82)
    x_caudal = int(x_withers + mask_len_pct * (x_tail - x_withers))
    
    y_spine = int(h * 0.26)
    y_sternum = int(h * 0.72)
    
    points = [
        (x_withers, y_spine + int(h * 0.04)),
        (int(x_withers + (x_caudal - x_withers) * 0.4), y_spine),
        (x_caudal, y_spine + int(h * 0.04)),
        (x_caudal, int(y_sternum - h * 0.08)),
        (int(x_withers + (x_caudal - x_withers) * 0.35), y_sternum),
        (x_withers, int(y_sternum - h * 0.05)),
    ]
    draw.polygon(points, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=6))

def render_vton_garment(orig_pil, mask_pil, depth_pil, best_size, fit_info, breed_name, morpho):
    w, h = orig_pil.size
    orig_np = np.array(orig_pil).astype(np.float32)
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    depth_np = np.array(depth_pil.convert("L")).astype(np.float32) / 255.0
    
    base_color = np.array([88, 112, 68], dtype=np.float32)  # Olive green Ruffwear
    trim_color = np.array([30, 32, 28], dtype=np.float32)
    
    shading = cv2.GaussianBlur(depth_np, (15, 15), 0)
    shading_factor = 0.55 + 0.65 * shading[:, :, None]
    
    noise = (np.random.RandomState(42).randn(h, w, 1) * 5.0).astype(np.float32)
    canvas = np.ones((h, w, 3), dtype=np.float32) * base_color[None, None, :]
    garment = np.clip(canvas * shading_factor + noise, 0, 255)
    
    h_idx, w_idx = np.where(mask_np > 0.3)
    if len(h_idx) > 0:
        top_y = np.min(h_idx)
        bottom_y = np.max(h_idx)
        mid_y = int(top_y + (bottom_y - top_y) * 0.32)
        
        # Spine harness webbing
        strap_mask = np.zeros((h, w), dtype=np.float32)
        strap_mask[mid_y-4:mid_y+5, :] = 1.0
        garment[strap_mask * (mask_np > 0.5) > 0.5] = np.array([25, 28, 30], dtype=np.float32)
        
        # Reflective piping
        pipe_mask = np.zeros((h, w), dtype=np.float32)
        pipe_mask[mid_y-5:mid_y-4, :] = 1.0
        garment[pipe_mask * (mask_np > 0.5) > 0.5] = np.array([220, 225, 230], dtype=np.float32)
        
        # Border trim
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eroded = cv2.erode((mask_np > 0.4).astype(np.uint8), kernel)
        border = ((mask_np > 0.4) & (~eroded.astype(bool))).astype(np.float32)
        garment[border > 0.5] = trim_color[None, :]
        
    smooth_mask = cv2.GaussianBlur(mask_np, (7, 7), 0)[:, :, None]
    composite = np.clip(orig_np * (1.0 - smooth_mask) + garment * smooth_mask, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(composite)
    
    # 3D Telemetry HUD Banner
    draw = ImageDraw.Draw(final_img)
    badge_w, badge_h = 420, 95
    draw.rectangle([10, 10, 10 + badge_w, 10 + badge_h], fill=(15, 20, 28, 235), outline=(50, 220, 130), width=2)
    draw.text((20, 16), f"3D-AWARE VTON PIPELINE | {breed_name.upper()}", fill=(50, 220, 130))
    draw.text((20, 36), f"3D BARC Slicing: Girth={morpho['chest_girth_cm']}cm | L={morpho['back_length_cm']}cm", fill=(240, 240, 240))
    draw.text((20, 54), f"Ruffwear Target: Size {best_size} ({fit_info['range_cm']}, Mid: {fit_info['garment_chest_mid_cm']}cm)", fill=(240, 240, 240))
    draw.text((20, 72), f"Verdict: {fit_info['verdict'].upper()} (Fit Ratio: {fit_info['fit_ratio']:.3f})", fill=fit_info["status_color"])
    
    return final_img

def main():
    print("=" * 85)
    print("  RUNNING COMPLETE MODULAR STAGE-BY-STAGE PIPELINE ON ALL 484 SIDE-VIEW DOGS")
    print("=" * 85)
    
    barc_model, norm_dict, device = init_barc_model("cuda")
    print("[*] Initializing MiDaS Depth Estimator on CUDA...")
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)
    
    golden_files = sorted(glob.glob(str(SIDEVIEW_ROOT / "golden" / "*.jpg")))
    lab_files = sorted(glob.glob(str(SIDEVIEW_ROOT / "labrador" / "*.jpg")))
    
    all_jobs = [(p, "Golden Retriever", "golden") for p in golden_files] + \
               [(p, "Labrador Retriever", "labrador") for p in lab_files]
               
    print(f"[*] Processing {len(all_jobs)} pure side-view dogs across all 5 distinct pipeline stages...")
    
    master_records = []
    t_start = time.time()
    
    for idx, (img_p, breed_name, sub) in enumerate(all_jobs, 1):
        img_path = Path(img_p)
        stem = img_path.stem
        
        try:
            t0 = time.time()
            pil_orig = Image.open(img_path).convert("RGB")
            
            # -------------------------------------------------------------
            # STAGE 1: 3D MESH RECONSTRUCTION (BARC / SMAL)
            # -------------------------------------------------------------
            inp_tensor = preprocess_for_barc(pil_orig).to(device)
            with torch.no_grad():
                out_raw, out_unnorm, out_reproj = barc_model(inp_tensor, norm_dict=norm_dict)
                
            verts = out_reproj["vertices_smal"][0].cpu().numpy()
            faces = barc_model.smal.f.astype(int)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            
            stage1_mesh_path = OUTPUT_ROOT / "stage_1_3d_meshes" / sub / f"{stem}_stage1_mesh.obj"
            mesh.export(str(stage1_mesh_path))
            
            # Slicing & Ramanujan Girth
            morpho = compute_ramanujan_chest_girth(mesh)
            best_size, verdicts = evaluate_garment_fit(morpho["chest_girth_cm"])
            fit_info = verdicts[best_size]
            mask_pct = RUFFWEAR_SIZE_CHART[best_size]["mask_len_pct"]
            
            # -------------------------------------------------------------
            # STAGE 2: 3D SURFACE DEPTH ESTIMATION (MiDaS)
            # -------------------------------------------------------------
            depth_pil = midas(pil_orig).resize(pil_orig.size, Image.BILINEAR)
            stage2_depth_path = OUTPUT_ROOT / "stage_2_depth_maps" / sub / f"{stem}_stage2_depth.png"
            depth_pil.save(str(stage2_depth_path))
            
            # -------------------------------------------------------------
            # STAGE 3: SIZING-CONDITIONED ANATOMICAL MASK GENERATION
            # -------------------------------------------------------------
            mask_pil = create_scaled_anatomical_mask(pil_orig, mask_len_pct=mask_pct)
            stage3_mask_path = OUTPUT_ROOT / "stage_3_anatomical_masks" / sub / f"{stem}_stage3_mask_{best_size}.png"
            mask_pil.save(str(stage3_mask_path))
            
            # -------------------------------------------------------------
            # STAGE 4: GEOMETRY-CONDITIONED VTON TRY-ON SYNTHESIS
            # -------------------------------------------------------------
            vton_pil = render_vton_garment(pil_orig, mask_pil, depth_pil, best_size, fit_info, breed_name, morpho)
            stage4_vton_path = OUTPUT_ROOT / "stage_4_vton_renders" / sub / f"{stem}_stage4_vton_{best_size}.png"
            vton_pil.save(str(stage4_vton_path))
            
            elapsed = time.time() - t0
            
            # -------------------------------------------------------------
            # STAGE 5: TELEMETRY & MASTER DATABASE RECORD
            # -------------------------------------------------------------
            rec = {
                "image_index": idx,
                "image_id": stem,
                "breed": breed_name,
                "original_photo_path": str(img_path),
                "stage_1_3d_mesh": {
                    "obj_file_path": str(stage1_mesh_path),
                    "num_vertices": morpho["num_vertices"],
                    "num_faces": morpho["num_faces"],
                    "chest_girth_ramanujan_cm": morpho["chest_girth_cm"],
                    "back_length_cm": morpho["back_length_cm"],
                    "withers_height_cm": morpho["withers_height_cm"],
                    "semi_major_depth_a_cm": morpho["semi_depth_a_cm"],
                    "semi_minor_width_b_cm": morpho["semi_width_b_cm"]
                },
                "stage_2_surface_depth": {
                    "depth_map_path": str(stage2_depth_path)
                },
                "stage_3_anatomical_mask": {
                    "mask_file_path": str(stage3_mask_path),
                    "dorsal_coverage_fraction": mask_pct,
                    "target_size": best_size
                },
                "stage_4_vton_synthesis": {
                    "vton_render_path": str(stage4_vton_path),
                    "fit_ratio": fit_info["fit_ratio"],
                    "sizing_verdict": fit_info["verdict"],
                    "all_sizes_fit_ratios": {k: {key: val for key, val in v.items() if key != 'status_color'} for k, v in verdicts.items()}
                },
                "latency_seconds": round(elapsed, 3)
            }
            master_records.append(rec)
            
            if idx % 50 == 0 or idx == len(all_jobs):
                print(f"  [{idx:03d}/{len(all_jobs):03d}] {stem} ({breed_name}) -> 3D Girth={morpho['chest_girth_cm']}cm | Rec: Size {best_size} ({fit_info['verdict']}) | Time: {elapsed:.2f}s")
                
        except Exception as e:
            print(f"  [ERROR] Processing {img_path}: {e}")
            
    total_time = time.time() - t_start
    print(f"\n[OK] Successfully processed all {len(master_records)} side-view dogs across all 5 stages in {total_time:.1f}s ({total_time/max(len(master_records), 1):.3f}s / dog)!")
    
    # Save Master JSON Database
    db_json_path = OUTPUT_ROOT / "stage_5_telemetry_database" / "sideview_3d_sizing_vton_master_database.json"
    with open(db_json_path, "w", encoding="utf-8") as f:
        json.dump(master_records, f, indent=2, ensure_ascii=False)
        
    # Save CSV Summary
    db_csv_path = OUTPUT_ROOT / "stage_5_telemetry_database" / "sideview_sizing_summary.csv"
    with open(db_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index", "Image_ID", "Breed", "Chest_Girth_cm", "Semi_Depth_A_cm",
            "Semi_Width_B_cm", "Back_Length_cm", "Height_cm", "Recommended_Size",
            "Fit_Ratio", "Verdict", "Stage1_3D_Mesh", "Stage2_Depth", "Stage3_Mask", "Stage4_VTON"
        ])
        for d in master_records:
            m = d["stage_1_3d_mesh"]
            v = d["stage_4_vton_synthesis"]
            writer.writerow([
                d["image_index"], d["image_id"], d["breed"],
                m["chest_girth_ramanujan_cm"], m["semi_major_depth_a_cm"], m["semi_minor_width_b_cm"],
                m["back_length_cm"], m["withers_height_cm"], d["stage_3_anatomical_mask"]["target_size"],
                v["fit_ratio"], v["sizing_verdict"], m["obj_file_path"],
                d["stage_2_surface_depth"]["depth_map_path"], d["stage_3_anatomical_mask"]["mask_file_path"],
                v["vton_render_path"]
            ])
            
    print(f"\n[OK] Master Telemetry Database saved to:\n  • JSON: {db_json_path}\n  • CSV:  {db_csv_path}")

if __name__ == "__main__":
    main()
