#!/usr/bin/env python3
import os
import sys
import json
import glob
import time
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import cv2
import torch
import trimesh

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BARC_ROOT = Path("C:/Users/Student3/Desktop/BARC")
REPO_DIR = Path("C:/Users/Student3/Desktop/ccbd_dogvton/3d-canine-sizing-vton")
sys.path.insert(0, str(BARC_ROOT / "src"))
sys.path.insert(0, str(REPO_DIR))

from configs.barc_cfg_defaults import update_cfg_global_with_yaml, get_cfg_global_updated
from configs.data_info import COMPLETE_DATA_INFO_24
from configs.dataset_path_configs import STANEXT_RELATED_DATA_ROOT_DIR
from combined_model.model_shape_v7 import ModelImageTo3d_withshape_withproj
from vton_controlnet import get_depth_map, load_controlnet_pipeline

DATASET_ROOT = Path("D:/canine-dog-dataset/dataset/raw")
OUTPUT_ROOT = Path("D:/barc_3d_full_dataset_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "golden" / "3d_meshes").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "golden" / "vton_images").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "labrador" / "3d_meshes").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "labrador" / "vton_images").mkdir(parents=True, exist_ok=True)

RUFFWEAR_SIZE_CHART = {
    "XXS": {"chest_min": 33.0, "chest_max": 43.0, "chest_mid": 38.0, "back_len_cm": 30.0, "mask_len_pct": 0.50},
    "XS":  {"chest_min": 43.0, "chest_max": 56.0, "chest_mid": 49.5, "back_len_cm": 38.0, "mask_len_pct": 0.60},
    "S":   {"chest_min": 56.0, "chest_max": 69.0, "chest_mid": 62.5, "back_len_cm": 48.0, "mask_len_pct": 0.72},
    "M":   {"chest_min": 69.0, "chest_max": 81.0, "chest_mid": 75.0, "back_len_cm": 58.0, "mask_len_pct": 0.85},
    "L":   {"chest_min": 81.0, "chest_max": 91.0, "chest_mid": 86.0, "back_len_cm": 68.0, "mask_len_pct": 0.95},
    "XL":  {"chest_min": 91.0, "chest_max": 107.0, "chest_mid": 99.0, "back_len_cm": 75.0, "mask_len_pct": 1.00},
}

def load_breed_names():
    dict_path = os.path.join(STANEXT_RELATED_DATA_ROOT_DIR, "StanExt_breed_dict_v2.json")
    if os.path.exists(dict_path):
        with open(dict_path, "r") as f:
            data = json.load(f)
            if "dict_breed_index_to_breed_name" in data:
                return data["dict_breed_index_to_breed_name"]
            elif "breed_list" in data:
                return {str(i): b for i, b in enumerate(data["breed_list"])}
            return data
    return {}

def init_barc_model(device_str="cuda"):
    checkpoint_path = BARC_ROOT / "checkpoint" / "barc_complete" / "model_best.pth.tar"
    cfg_file = BARC_ROOT / "src" / "configs" / "barc_cfg_visualization.yaml"
    update_cfg_global_with_yaml(str(cfg_file))
    cfg = get_cfg_global_updated()
    
    device_obj = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"[*] Initializing BARC Model on device: {device_obj}")
    
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

def preprocess_for_barc(img_path, target_size=256):
    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size
    max_side = max(w, h)
    
    padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
    start_x = (max_side - w) // 2
    start_y = (max_side - h) // 2
    padded.paste(pil_img, (start_x, start_y))
    
    resized = padded.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    img_torch = torch.from_numpy(img_np.transpose(2, 0, 1))
    
    rgb_mean = torch.tensor(COMPLETE_DATA_INFO_24.rgb_mean, dtype=torch.float32).view(3, 1, 1)
    rgb_std = torch.tensor(COMPLETE_DATA_INFO_24.rgb_stddev, dtype=torch.float32).view(3, 1, 1)
    inp = (img_torch - rgb_mean) / rgb_std
    return inp.unsqueeze(0), pil_img

def compute_3d_morphometrics(mesh, scale_factor_cm=75.0):
    bounds = mesh.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]
    y_min, y_max = bounds[0][1], bounds[1][1]
    
    slice_z = z_min + (z_max - z_min) * 0.58
    slice_chest = mesh.section(plane_origin=[0, 0, slice_z], plane_normal=[0, 0, 1])
    if slice_chest is not None and hasattr(slice_chest, "length") and slice_chest.length > 0:
        raw_chest_perimeter = float(slice_chest.length)
    else:
        raw_chest_perimeter = (bounds[1][0] - bounds[0][0] + bounds[1][1] - bounds[0][1]) * 1.5
        
    verts = mesh.vertices
    withers_pt = verts[np.argmax(verts[:, 1])]
    tail_base_pt = verts[np.argmin(verts[:, 2])]
    raw_back_len = np.linalg.norm(withers_pt - tail_base_pt)
    raw_height = y_max - y_min
    
    chest_girth_cm = round(raw_chest_perimeter * scale_factor_cm, 1)
    back_len_cm = round(raw_back_len * scale_factor_cm * 0.85, 1)
    height_cm = round(raw_height * scale_factor_cm, 1)
    
    return {
        "chest_girth_cm": float(chest_girth_cm),
        "back_length_cm": float(back_len_cm),
        "height_cm": float(height_cm),
        "raw_mesh_bounds": mesh.bounds.tolist(),
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

def main():
    print("=" * 80)
    print("  RUNNING FULL BARC 3D MESH + VTON PIPELINE ON ALL CANINE IMAGES (RTX 4090)")
    print("=" * 80)
    
    barc_model, norm_dict, device = init_barc_model("cuda")
    vton_pipe = load_controlnet_pipeline()
    breed_dict = load_breed_names()
    
    golden_images = sorted(glob.glob(str(DATASET_ROOT / "golden" / "*.jpg")))
    lab_images = sorted(glob.glob(str(DATASET_ROOT / "labrador" / "*.jpg")))
    
    all_image_jobs = [(p, "Golden Retriever", OUTPUT_ROOT / "golden") for p in golden_images] + \
                     [(p, "Labrador Retriever", OUTPUT_ROOT / "labrador") for p in lab_images]
                     
    print(f"[*] Total dataset images to process: {len(all_image_jobs)} ({len(golden_images)} Golden + {len(lab_images)} Labrador)")
    
    all_database_records = []
    t_start = time.time()
    
    for idx, (img_p, breed_name, out_dir) in enumerate(all_image_jobs, 1):
        img_path = Path(img_p)
        stem = img_path.stem
        
        try:
            # 1. BARC Forward Pass
            inp_tensor, pil_orig = preprocess_for_barc(img_path)
            inp_tensor = inp_tensor.float().to(device)
            
            with torch.no_grad():
                out_raw, out_unnorm, out_reproj = barc_model(inp_tensor, norm_dict=norm_dict)
                
            vertices = out_reproj["vertices_smal"][0].cpu().numpy()
            faces = barc_model.smal.f.astype(int)
            
            # Export BARC 3D Mesh (.obj)
            obj_filename = f"{stem}_3d_mesh.obj"
            obj_path = out_dir / "3d_meshes" / obj_filename
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.export(str(obj_path))
            
            # Extract SMAL parameters
            breed_tensor = out_reproj.get("dog_breed", None)
            breed_idx = int(torch.argmax(breed_tensor[0]).cpu().numpy()) if breed_tensor is not None else 0
            predicted_breed = breed_dict.get(str(breed_idx), f"Breed_{breed_idx}")
            
            # 2. Compute True 3D Mesh Morphometrics
            morphometrics = compute_3d_morphometrics(mesh)
            best_size, verdicts = evaluate_garment_fit(morphometrics["chest_girth_cm"])
            fit_info = verdicts[best_size]
            mask_pct = RUFFWEAR_SIZE_CHART[best_size]["mask_len_pct"]
            
            # 3. Diffusion Prompt Conditioning
            if fit_info["verdict"] == "Good Fit":
                fit_prompt = "perfectly fitting dog jacket coat, comfortable natural drape from withers to loin"
            elif fit_info["verdict"] == "Tight":
                fit_prompt = "tight fitting harness vest, taut fabric pulling at seams, short fit ending mid-back"
            else:
                fit_prompt = "loose oversized dog jacket coat, sagging fabric folds, extended drape covering hips"
                
            pos_prompt = (
                f"photo of a {breed_name.lower()} wearing an olive green Ruffwear dog jacket coat vest, size {best_size}, {fit_prompt}, "
                "saddle-style dog coat, belly strap with black buckle, D-ring handle on back, "
                "olive army green waterproof fabric, black trim edges, reflective piping, "
                "jacket draped naturally over curved dog back, fur visible at collar and legs, "
                "realistic photographic quality, natural lighting, photorealistic 8k photo"
            )
            neg_prompt = "cartoon, painting, illustration, anime, blurry, distorted, human clothes, deformed anatomy, text, low quality"
            
            # 4. Generate VTON Image
            SD_SIZE = (512, 512)
            dog_sd = pil_orig.resize(SD_SIZE, Image.LANCZOS)
            depth_map = get_depth_map(pil_orig, SD_SIZE)
            
            # Render VTON
            t_vton0 = time.time()
            vton_out = vton_pipe(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                image=dog_sd,
                mask_image=depth_map, # Depth-conditioned torso region
                control_image=depth_map,
                height=SD_SIZE[1],
                width=SD_SIZE[0],
                num_inference_steps=28,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.45,
                strength=0.85
            )
            gen_img = vton_out.images[0]
            vton_elapsed = time.time() - t_vton0
            
            # Overlay 3D Sizing & BARC HUD Banner
            final_img = gen_img.resize(pil_orig.size, Image.LANCZOS)
            draw = ImageDraw.Draw(final_img)
            badge_w, badge_h = 390, 95
            draw.rectangle([10, 10, 10 + badge_w, 10 + badge_h], fill=(18, 22, 28, 230), outline=(50, 180, 100), width=2)
            draw.text((20, 16), f"BARC 3D CANINE SIZING | {breed_name.upper()}", fill=(50, 220, 130))
            draw.text((20, 36), f"BARC 3D Mesh: L={morphometrics['back_length_cm']}cm | Girth={morphometrics['chest_girth_cm']}cm", fill=(240, 240, 240))
            draw.text((20, 54), f"Garment: Ruffwear Size {best_size} ({fit_info['range_cm']}, Mid: {fit_info['garment_chest_mid_cm']}cm)", fill=(240, 240, 240))
            draw.text((20, 72), f"Verdict: {fit_info['verdict'].upper()} (Fit Ratio: {fit_info['fit_ratio']:.3f})", fill=fit_info["status_color"])
            
            vton_filename = f"{stem}_vton_{best_size}.png"
            vton_path = out_dir / "vton_images" / vton_filename
            final_img.save(str(vton_path), format="PNG")
            
            record = {
                "image_index": idx,
                "image_id": stem,
                "breed": breed_name,
                "predicted_barc_breed": predicted_breed,
                "original_file": str(img_path),
                "barc_3d_measurements": {
                    "chest_girth_cm": morphometrics["chest_girth_cm"],
                    "back_length_cm": morphometrics["back_length_cm"],
                    "height_cm": morphometrics["height_cm"],
                    "num_3d_vertices": morphometrics["num_vertices"],
                    "num_3d_faces": morphometrics["num_faces"],
                    "source": "BARC SMAL 3,889-Vertex 3D Mesh Planar Intersection Slicing"
                },
                "sizing_evaluation": {
                    "recommended_size": best_size,
                    "fit_ratio": fit_info["fit_ratio"],
                    "pct_difference": fit_info["pct_diff"],
                    "sizing_verdict": fit_info["verdict"],
                    "all_sizes_fit_ratios": {k: {key: val for key, val in v.items() if key != 'status_color'} for k, v in verdicts.items()}
                },
                "diffusion_model_prompts": {
                    "positive_prompt": pos_prompt,
                    "negative_prompt": neg_prompt,
                    "guidance_scale": 7.5,
                    "strength": 0.85,
                    "controlnet_conditioning_scale": 0.45,
                    "num_inference_steps": 28,
                    "mask_length_percentage": int(mask_pct * 100)
                },
                "output_files": {
                    "barc_3d_mesh_obj": str(obj_path),
                    "vton_image_png": str(vton_path)
                }
            }
            all_database_records.append(record)
            
            if idx % 50 == 0 or idx == len(all_image_jobs):
                print(f"  [{idx:03d}/{len(all_image_jobs):03d}] {stem} ({breed_name}) -> 3D Girth={morphometrics['chest_girth_cm']}cm | Rec: Size {best_size} ({fit_info['verdict']}) | Mesh: 3,889 verts")
                
        except Exception as e:
            print(f"  [ERROR] Processing {img_path}: {e}")
            
    total_time = time.time() - t_start
    print(f"\n[✓] Finished BARC 3D + VTON for all {len(all_database_records)} images in {total_time:.1f}s ({total_time/max(len(all_database_records), 1):.3f}s / dog)!")
    
    # Save Master Database
    master_json_path = OUTPUT_ROOT / "master_barc_3d_vton_all_915_dogs.json"
    with open(master_json_path, "w") as f:
        json.dump(all_database_records, f, indent=2)
        
    print(f"\n[✓] Master 915-Dog Database saved to:\n    {master_json_path}")

if __name__ == "__main__":
    main()
