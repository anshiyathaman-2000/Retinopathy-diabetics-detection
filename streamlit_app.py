import os
import cv2
import numpy as np
import pandas as pd
import torch
import streamlit as st
from PIL import Image

from model import get_model
from preprocessing import apply_clahe, multi_scale_retinex

# Page Config
st.set_page_config(
    page_title="RetiHybrid-CTL Portal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Dataset/Config Detection
archive2_dir = os.path.join(BASE_DIR, "archive 2/retino")
CSV_PATH = os.path.join(BASE_DIR, "Evaluation_Set/RFMiD_Validation_Labels.csv")

if os.path.exists(archive2_dir):
    is_archive2 = True
    label_cols = ["Disease_Risk", "DR"]
    VALIDATION_IMG_DIR = os.path.join(archive2_dir, "valid")
else:
    is_archive2 = False
    VALIDATION_IMG_DIR = os.path.join(BASE_DIR, "Evaluation_Set/Validation")
    if os.path.exists(CSV_PATH):
        df_labels = pd.read_csv(CSV_PATH)
        label_cols = df_labels.columns[1:].tolist()
    else:
        label_cols = ["Disease_Risk"] + [f"Class_{i}" for i in range(1, 46)]

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

# Sidebar - Model Selection
st.sidebar.image("https://img.icons8.com/color/96/ophthalmology.png", width=80)
st.sidebar.markdown("### Model Configuration")

model_options = {
    "hybrid": "CNN-Transformer-LSTM (Proposed)",
    "resnet50": "ResNet50 Baseline",
    "vit": "Vision Transformer (ViT) Baseline",
    "cnn_transformer": "CNN-Transformer Hybrid",
    "cnn_lstm": "CNN-LSTM Hybrid",
    "cnn_inception": "CNN Dual Backbone"
}

selected_model_id = st.sidebar.selectbox(
    "Choose Model Architecture",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# Load Model Dynamically with Class Count Detection
def detect_classes_from_checkpoint(checkpoint):
    if "classifier.bias" in checkpoint:
        return checkpoint["classifier.bias"].shape[0]
    elif "fc.1.bias" in checkpoint:
        return checkpoint["fc.1.bias"].shape[0]
    else:
        bias_keys = [k for k in checkpoint.keys() if "bias" in k and ("classifier" in k or "fc" in k)]
        if len(bias_keys) > 0:
            return checkpoint[bias_keys[0]].shape[0]
    return 46  # Default fallback

uploaded_weights = st.sidebar.file_uploader("Upload .pth weights manually", type=["pth"])

weight_filename = f"best_model_{selected_model_id}.pth"
weight_path = os.path.join(STATIC_DIR, weight_filename)
if selected_model_id == "hybrid" and not os.path.exists(weight_path):
    legacy_path = os.path.join(STATIC_DIR, "best_model.pth")
    if os.path.exists(legacy_path):
        weight_path = legacy_path

checkpoint = None
source_name = ""

if uploaded_weights is not None:
    try:
        checkpoint = torch.load(uploaded_weights, map_location="cpu")
        source_name = f"Uploaded File: {uploaded_weights.name}"
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded file: {e}")
elif os.path.exists(weight_path):
    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
        source_name = os.path.basename(weight_path)
    except Exception as e:
        st.sidebar.error(f"Error loading local weights {os.path.basename(weight_path)}: {e}")

if checkpoint is not None:
    try:
        detected_classes = detect_classes_from_checkpoint(checkpoint)
        model = get_model(selected_model_id, num_classes=detected_classes)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Override label columns dynamically based on detected model classes
        if detected_classes == 2:
            label_cols = ["Disease_Risk", "DR"]
        else:
            if os.path.exists(CSV_PATH):
                df_labels = pd.read_csv(CSV_PATH)
                label_cols = df_labels.columns[1:].tolist()
            else:
                label_cols = ["Disease_Risk"] + [f"Class_{i}" for i in range(1, detected_classes)]
                
        st.sidebar.success(f"✓ Loaded weights: {source_name} ({detected_classes} classes)")
    except Exception as e:
        st.sidebar.error(f"Failed to load weights: {e}")
        model = get_model(selected_model_id, num_classes=len(label_cols))
        model.eval()
else:
    st.sidebar.warning("⚠️ No weights found. Running with initialized weights.")
    model = get_model(selected_model_id, num_classes=len(label_cols))
    model.eval()

# Device check
device_str = "Apple Silicon MPS" if torch.backends.mps.is_available() else ("CUDA GPU" if torch.cuda.is_available() else "CPU")
st.sidebar.info(f"Hardware Acceleration: **{device_str}**")

# Main Portal Layout
st.markdown("<h1 class='main-title'>RetiHybrid-CTL Diagnostic Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Joint Ophthalmic Multi-Label Retinal Disease Diagnosis & Fundus Enhancement System</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["👁️ Diagnostic Pipeline", "📊 Performance Metrics", "📄 Project Documentation"])

with tab1:
    st.markdown("### Step 1: Input Fundus Image")
    input_source = st.radio("Choose Input Source", ["Upload Custom Image", "Select Validation Dataset Sample"], horizontal=True)
    
    image_to_process = None
    ground_truth = None
    
    if input_source == "Upload Custom Image":
        uploaded_file = st.file_uploader("Upload fundus image (JPG, PNG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        # Load sample examples dynamically
        examples = []
        if is_archive2 and os.path.exists(VALIDATION_IMG_DIR):
            dr_dir = os.path.join(VALIDATION_IMG_DIR, "DR")
            nodr_dir = os.path.join(VALIDATION_IMG_DIR, "No_DR")
            dr_files = sorted([f for f in os.listdir(dr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:15] if os.path.exists(dr_dir) else []
            nodr_files = sorted([f for f in os.listdir(nodr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:15] if os.path.exists(nodr_dir) else []
            for f in dr_files:
                examples.append({"name": f"DR/{f}", "path": os.path.join(dr_dir, f), "gt": {"Disease_Risk": 1, "DR": 1}})
            for f in nodr_files:
                examples.append({"name": f"No_DR/{f}", "path": os.path.join(nodr_dir, f), "gt": {"Disease_Risk": 0, "DR": 0}})
        elif os.path.exists(CSV_PATH) and os.path.exists(VALIDATION_IMG_DIR):
            df = pd.read_csv(CSV_PATH)
            for idx, row in df.head(30).iterrows():
                img_name = f"{int(row['ID'])}.png"
                img_path = os.path.join(VALIDATION_IMG_DIR, img_name)
                if os.path.exists(img_path):
                    gt_dict = {col: int(row[col]) for col in label_cols}
                    examples.append({"name": img_name, "path": img_path, "gt": gt_dict})
        
        if len(examples) > 0:
            example_names = [x["name"] for x in examples]
            selected_example_name = st.selectbox("Select Sample Image", example_names)
            selected_example = next(x for x in examples if x["name"] == selected_example_name)
            image_to_process = cv2.imread(selected_example["path"])
            ground_truth = selected_example["gt"]
        else:
            st.info("No local validation samples found. Please upload a custom image.")

    if image_to_process is not None:
        st.markdown("### Step 2: Fundus Enhancement & Preprocessing")
        
        # Preprocessing Steps
        resized = cv2.resize(image_to_process, (256, 256), interpolation=cv2.INTER_AREA)
        clahe_img = apply_clahe(resized)
        msr_img = multi_scale_retinex(clahe_img)
        
        # Display side-by-side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), caption="Original Fundus Image (Resized)", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB), caption="LAB CLAHE Enhanced", use_column_width=True)
        with col3:
            st.image(cv2.cvtColor(msr_img, cv2.COLOR_BGR2RGB), caption="Multi-Scale Retinex (MSR) Output", use_column_width=True)
            
        # Diagnosis Button
        if st.button("👁️ Run Disease Diagnosis", type="primary"):
            with st.spinner("Analyzing fundus scans..."):
                # Prepare tensor
                img_rgb = cv2.cvtColor(msr_img, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_normalized = (img_normalized - mean) / std
                img_tensor = img_normalized.transpose(2, 0, 1)
                img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    
                # Format predictions
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
                
                # Split Disease Risk and specific classes
                disease_risk_pred = [p for p in predictions if p["label"] == "Disease_Risk"][0]
                other_preds = [p for p in predictions if p["label"] != "Disease_Risk"]
                other_preds.sort(key=lambda x: x["probability"], reverse=True)
                
                # Display Results
                st.markdown("---")
                st.markdown("### Diagnosis Summary")
                
                risk_prob = disease_risk_pred["probability"]
                risk_color = "#ef4444" if risk_prob >= 0.5 else "#10b981"
                
                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    st.markdown(f"""
                        <div class='metric-card' style='border-top: 5px solid {risk_color};'>
                            <h4 style='margin-top:0;'>Disease Risk Index</h4>
                            <h2 style='color: {risk_color}; margin: 0;'>{risk_prob:.2%}</h2>
                            <p style='color:#64748b; font-size:0.9rem; margin-top:5px;'>
                                Status: <b>{'High Risk' if risk_prob >= 0.5 else 'Low/Normal Risk'}</b>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                with r_col2:
                    active_diseases = [p["fullname"] for p in other_preds if p["prediction"] == 1]
                    if len(active_diseases) > 0:
                        st.markdown(f"#### Detected Pathologies:")
                        for d in active_diseases:
                            st.error(f"⚠️ **{d}** detected")
                    else:
                        st.success("✅ No specific pathologies detected with threshold >= 0.50")
                
                # Details Table
                st.markdown("#### Full Clinical Classification Report")
                
                report_data = []
                for p in [disease_risk_pred] + other_preds:
                    status = "🔴 Positive" if p["prediction"] == 1 else "🟢 Negative"
                    gt_status = "N/A"
                    if p["ground_truth"] is not None:
                        gt_status = "🔴 Positive" if p["ground_truth"] == 1 else "🟢 Negative"
                    
                    report_data.append({
                        "Label": p["label"],
                        "Full Disease Name": p["fullname"],
                        "Confidence Score": f"{p['probability']:.4%}",
                        "Classification Decision": status,
                        "Ground Truth": gt_status
                    })
                    
                st.table(pd.DataFrame(report_data))

with tab2:
    st.markdown("### Model Performance Analysis")
    
    # Load stats from static folder
    m_name = selected_model_id.lower().strip()
    stats_path = os.path.join(STATIC_DIR, f"evaluation_results_{m_name}.txt")
    if not os.path.exists(stats_path) and m_name == "hybrid":
        stats_path = os.path.join(STATIC_DIR, "evaluation_results.txt")
        
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats_content = f.read()
            st.text_area("Evaluation Report", stats_content, height=250)
    else:
        st.warning("No evaluation metrics text file found for this model.")
        
    # Visual metrics curves
    st.markdown("### Visual Performance Curves")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        loss_fn = f"loss_plot_{m_name}.png" if os.path.exists(os.path.join(STATIC_DIR, f"loss_plot_{m_name}.png")) else "loss_plot.png"
        loss_path = os.path.join(STATIC_DIR, loss_fn)
        if os.path.exists(loss_path):
            st.image(loss_path, caption=f"Training & Validation Loss Curves ({model_options[selected_model_id]})", use_column_width=True)
        else:
            st.info("Loss plot not found.")
            
    with col_img2:
        confusion_path = os.path.join(STATIC_DIR, "confusion_matrix.png")
        if os.path.exists(confusion_path):
            st.image(confusion_path, caption="Confusion Matrix (Validation)", use_column_width=True)
        else:
            st.info("Confusion matrix not found.")

with tab3:
    st.markdown("### Implementation Methodology")
    
    paper_path = os.path.join(BASE_DIR, "implementation_paper.md")
    if os.path.exists(paper_path):
        with open(paper_path, "r") as f:
            paper_content = f.read()
            st.markdown(paper_content)
    else:
        st.info("Implementation paper markdown file not found.")
