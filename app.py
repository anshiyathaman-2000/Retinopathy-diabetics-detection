import os
import shutil
import uuid
import numpy as np
import pandas as pd
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from typing import Optional

from model import HybridCNNTransformerLSTM
from preprocessing import apply_clahe, multi_scale_retinex

# Create FastAPI app
app = FastAPI(title="RetiHybrid-CTL API", description="Ophthalmic Multi-Label Retinal Disease Diagnosis API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper"
CSV_PATH = os.path.join(BASE_DIR, "Evaluation_Set/RFMiD_Validation_Labels.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_RUNS_DIR = os.path.join(STATIC_DIR, "temp_runs")


# Ensure directories exist
os.makedirs(TEMP_RUNS_DIR, exist_ok=True)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("FastAPI using Apple Silicon MPS GPU Acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("FastAPI using CUDA GPU Acceleration")
else:
    device = torch.device("cpu")
    print("FastAPI using CPU")

# Dynamic detection of checkpoint classes and configuration mapping
model_path = os.path.join(STATIC_DIR, "best_model.pth")
is_archive2 = False
label_cols = ["Disease_Risk"] + [f"Class_{i}" for i in range(1, 46)] # default fallback

if os.path.exists(model_path):
    try:
        # Load CPU version to inspect classifier shape
        checkpoint = torch.load(model_path, map_location="cpu")
        if "classifier.bias" in checkpoint:
            ckpt_classes = checkpoint["classifier.bias"].shape[0]
            if ckpt_classes == 2:
                is_archive2 = True
                label_cols = ["Disease_Risk", "DR"]
                print("FastAPI: Detected model checkpoint trained on archive 2 (2 classes).")
            else:
                print(f"FastAPI: Detected model checkpoint with {ckpt_classes} classes.")
                if os.path.exists(CSV_PATH):
                    df_labels = pd.read_csv(CSV_PATH)
                    label_cols = df_labels.columns[1:].tolist()
    except Exception as e:
        print(f"FastAPI: Error inspecting checkpoint: {e}")
else:
    # Fallback: check if archive 2 dataset exists
    archive2_dir = os.path.join(BASE_DIR, "archive 2/retino")
    if os.path.exists(archive2_dir):
        is_archive2 = True
        label_cols = ["Disease_Risk", "DR"]
        print("FastAPI: Model checkpoint not found. Defaulting to archive 2 config.")
    else:
        if os.path.exists(CSV_PATH):
            df_labels = pd.read_csv(CSV_PATH)
            label_cols = df_labels.columns[1:].tolist()

if is_archive2:
    VALIDATION_IMG_DIR = os.path.join(BASE_DIR, "archive 2/retino/valid")
else:
    VALIDATION_IMG_DIR = os.path.join(BASE_DIR, "Evaluation_Set/Validation")


# Abbreviation mapping for labels
DISEASE_MAP = {
    "Disease_Risk": "General Disease Risk",
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH": "Media Haze",
    "DN": "Drusen",
    "MYA": "Pathological Myopia",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation",
    "ERM": "Epiretinal Membrane",
    "LS": "Laser Scars",
    "MS": "Macular Scar",
    "CSR": "Central Serous Chorioretinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels",
    "AH": "Asteroid Hyalosis",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Shunt Vessels",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Pterygium",
    "RT": "Retinal Tear",
    "RS": "Retinitis",
    "CRS": "Chorioretinitis",
    "EDN": "Exudation",
    "RPEC": "Retinal Pigment Epithelium Changes",
    "MHL": "Macular Hole",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton Wool Spots",
    "CB": "Coloboma",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "MNF": "Myelinated Nerve Fibers",
    "HR": "Hemorrhage",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD": "Tilted Disc",
    "CME": "Cystoid Macular Edema",
    "PTCR": "Post-Traumatic Choroidal Rupture",
    "CF": "Choroidal Folds",
    "VH": "Vitreous Hemorrhage",
    "MCA": "Macroaneurysm",
    "VS": "Vasculitis",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ": "Plaque",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "CL": "Collateral Vessels"
}

# Initialize and load model weights
model = HybridCNNTransformerLSTM(num_classes=len(label_cols))
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded model weights.")
    except Exception as e:
        print(f"Error loading model weights: {e}. Starting with uninitialized weights.")
else:
    print(f"Warning: Model checkpoint not found at {model_path}. Please train first.")
model.to(device)
model.eval()

# Helper for processing and saving intermediate states
def process_and_save_steps(img_path: str, run_id: str):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
        
    # Step 1: Resize
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    original_fn = f"{run_id}_0_original.png"
    original_path = os.path.join(TEMP_RUNS_DIR, original_fn)
    cv2.imwrite(original_path, resized)
    
    # Step 2: LAB CLAHE
    clahe_img = apply_clahe(resized)
    clahe_fn = f"{run_id}_1_clahe.png"
    clahe_path = os.path.join(TEMP_RUNS_DIR, clahe_fn)
    cv2.imwrite(clahe_path, clahe_img)
    
    # Step 3: Multi-Scale Retinex
    msr_img = multi_scale_retinex(clahe_img)
    msr_fn = f"{run_id}_2_msr.png"
    msr_path = os.path.join(TEMP_RUNS_DIR, msr_fn)
    cv2.imwrite(msr_path, msr_img)
    
    return {
        "original": f"/static/temp_runs/{original_fn}",
        "clahe": f"/static/temp_runs/{clahe_fn}",
        "msr": f"/static/temp_runs/{msr_fn}",
        "processed_img": msr_img
    }

@app.get("/api/examples")
def get_examples():
    """
    Returns a list of interesting validation images and their ground truths to try out.
    """
    if is_archive2:
        examples = []
        dr_dir = os.path.join(VALIDATION_IMG_DIR, "DR")
        nodr_dir = os.path.join(VALIDATION_IMG_DIR, "No_DR")
        
        dr_files = []
        nodr_files = []
        if os.path.exists(dr_dir):
            dr_files = sorted([f for f in os.listdir(dr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if os.path.exists(nodr_dir):
            nodr_files = sorted([f for f in os.listdir(nodr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
        # Select first 25 of each to make it balanced
        selected_dr = [("DR", f) for f in dr_files[:25]]
        selected_nodr = [("No_DR", f) for f in nodr_files[:25]]
        raw_examples = selected_dr + selected_nodr
        
        for idx, (label_name, f) in enumerate(raw_examples):
            examples.append({
                "id": idx,
                "filename": f"{label_name}/{f}",
                "url": f"/validation_images/{label_name}/{f}",
                "disease_risk": 1 if label_name == "DR" else 0,
                "active_diseases": ["DR"] if label_name == "DR" else [],
                "ground_truth": {
                    "Disease_Risk": 1 if label_name == "DR" else 0,
                    "DR": 1 if label_name == "DR" else 0
                }
            })
        return examples

        
    if not os.path.exists(CSV_PATH):
        return []
    
    # Read the labels file
    df = pd.read_csv(CSV_PATH)
    
    # Pick a set of interesting examples that have at least one active disease
    # We want to represent multiple active categories (DR, ARMD, MH, ODC, etc.)
    # Disease_Risk is column 2 (index 1)
    examples = []
    
    # Sort or select rows with various diseases
    # Let's check some rows that have DR=1, ARMD=1, MH=1, ODC=1, and some with no disease (Disease_Risk=0)
    for idx, row in df.iterrows():
        img_id = int(row['ID'])
        img_name = f"{img_id}.png"
        img_path = os.path.join(VALIDATION_IMG_DIR, img_name)
        
        # Verify file exists
        if not os.path.exists(img_path):
            continue
            
        active = []
        for col in label_cols[1:]:  # Skip Disease_Risk for the disease list
            if row[col] == 1:
                active.append(col)
                
        # Group into a clean structure
        examples.append({
            "id": img_id,
            "filename": img_name,
            "url": f"/validation_images/{img_name}",
            "disease_risk": int(row['Disease_Risk']),
            "active_diseases": active,
            "ground_truth": {col: int(row[col]) for col in label_cols}
        })
        
        # Limit to 50 interesting samples to avoid overwhelming the frontend
        if len(examples) >= 50:
            break
            
    # Sort examples so that those with more active diseases appear first
    examples.sort(key=lambda x: len(x["active_diseases"]), reverse=True)
    return examples

@app.post("/api/predict")
async def predict(
    image_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Runs the full diagnosis pipeline:
    - Preprocessing (original -> CLAHE -> MSR)
    - Hybrid Model inference
    - Ground Truth comparisons (if validation image)
    """
    run_id = str(uuid.uuid4())
    temp_input_path = None
    
    try:
        # Determine source image
        if file is not None and file.filename != "":
            # Custom upload
            temp_input_path = os.path.join(TEMP_RUNS_DIR, f"{run_id}_input.png")
            with open(temp_input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            ground_truth = None
            img_id_display = file.filename
        elif image_id is not None:
            # Preloaded validation sample
            if is_archive2:
                examples = get_examples()
                example = next((x for x in examples if x["id"] == image_id), None)
                if example is None:
                    raise HTTPException(status_code=404, detail=f"Validation image index {image_id} not found.")
                
                img_subpath = example["filename"]
                temp_input_path = os.path.join(VALIDATION_IMG_DIR, img_subpath)
                ground_truth = example["ground_truth"]
                img_id_display = f"Validation Image #{image_id} ({os.path.basename(img_subpath)})"
            else:
                img_name = f"{image_id}.png"
                temp_input_path = os.path.join(VALIDATION_IMG_DIR, img_name)
                if not os.path.exists(temp_input_path):
                    raise HTTPException(status_code=404, detail=f"Validation image {image_id} not found.")
                
                # Fetch ground truth from CSV
                df = pd.read_csv(CSV_PATH)
                row = df[df['ID'] == image_id]
                if not row.empty:
                    row_data = row.iloc[0]
                    ground_truth = {col: int(row_data[col]) for col in label_cols}
                else:
                    ground_truth = None
                img_id_display = f"Validation Image #{image_id}"
        else:
            raise HTTPException(status_code=400, detail="Provide either an image file or a validation image ID.")
            
        # 1. Run and save preprocessing steps
        steps = process_and_save_steps(temp_input_path, run_id)
        
        # 2. Normalize and prepare for PyTorch model
        processed_img = steps["processed_img"]
        # BGR to RGB
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Tensor formatting: HWC -> CHW, unsqueeze batch
        img_tensor = img_normalized.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
        
        # 3. Model Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        # 4. Format Predictions
        predictions = []
        for i, col in enumerate(label_cols):
            prob = float(probs[i])
            pred_class = 1 if prob >= 0.5 else 0
            gt_val = ground_truth.get(col) if ground_truth else None
            
            predictions.append({
                "label": col,
                "fullname": DISEASE_MAP.get(col, col),
                "probability": prob,
                "prediction": pred_class,
                "ground_truth": gt_val
            })
            
        # Sort predictions by probability descending, keeping Disease_Risk at top
        disease_risk_pred = [p for p in predictions if p["label"] == "Disease_Risk"][0]
        other_preds = [p for p in predictions if p["label"] != "Disease_Risk"]
        other_preds.sort(key=lambda x: x["probability"], reverse=True)
        
        sorted_predictions = [disease_risk_pred] + other_preds
        
        return {
            "run_id": run_id,
            "image_id": img_id_display,
            "images": {
                "original": steps["original"],
                "clahe": steps["clahe"],
                "msr": steps["msr"]
            },
            "predictions": sorted_predictions,
            "ground_truth_available": ground_truth is not None
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    finally:
        # Clean up temporary uploaded input image if it exists
        if file is not None and temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except:
                pass

@app.get("/api/stats")
def get_stats():
    """
    Parses and returns evaluation results from evaluation_results.txt.
    """
    stats_path = os.path.join(STATIC_DIR, "evaluation_results.txt")
    if not os.path.exists(stats_path):
        return {"error": "Stats not found"}
        
    try:
        with open(stats_path, "r") as f:
            lines = f.readlines()
            
        res = {}
        for line in lines:
            line_str = line.strip()
            if ":" in line_str and not line_str.startswith("PER-CLASS"):
                parts = line_str.split(":")
                key = parts[0].strip()
                val = parts[1].strip()
                res[key] = val
        return res
    except Exception as e:
        return {"error": str(e)}

# Serve index.html at root "/"
@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Index.html not found</h1>"

# Serve implementation_paper.md
@app.get("/implementation_paper.md", response_class=HTMLResponse)
def get_paper():
    paper_path = os.path.join(BASE_DIR, "implementation_paper.md")
    if os.path.exists(paper_path):
        with open(paper_path, "r") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="Paper not found")

# Serve specific codebase files as plain text to avoid local file:// security blockages in browser
@app.get("/code/{filename}", response_class=PlainTextResponse)
def get_code(filename: str):
    allowed_files = ["preprocessing.py", "dataset.py", "model.py", "train_eval.py"]
    if filename not in allowed_files:
        raise HTTPException(status_code=403, detail="Access denied")
    
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="File not found")


# Mount static and validation directories
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/validation_images", StaticFiles(directory=VALIDATION_IMG_DIR), name="validation_images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
