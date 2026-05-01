import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
import pandas as pd
import random
import os
import requests

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Chest X-Ray Report Generator",
    page_icon="🩻",
    layout="wide"
)
st.title("Chest X-Ray Report Generator")
st.markdown("**Swin Transformer + IU Chest X-ray Templates · FYP**")

# ============================================================
# HUGGING FACE MODEL URL (for cloud deployment)
# ============================================================
MODEL_URL = "https://huggingface.co/MSA222/chest-xray-model/resolve/main/best_cls.pt"

# ============================================================
# DISEASE CLASSES
# ============================================================
DISEASES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
]

# ============================================================
# REAL IU CHEST X-RAY REPORT TEMPLATES (from indiana_reports.csv)
# ============================================================
REPORT_TEMPLATES = {
    "No Finding": [
        "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no signs of a pleural effusion. There is no evidence of pneumothorax. Impression: Normal chest x-ray.",
        "Lungs are clear without focal consolidation, effusion, or pneumothorax. Normal heart size. Cardiomediastinal silhouette is unremarkable. Impression: No acute cardiopulmonary abnormality.",
        "The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. No focal consolidation, pleural effusion, or pneumothorax identified. Impression: No acute cardiopulmonary process."
    ],
    "Pneumonia": [
        "There is focal airspace opacity consistent with pneumonia. Consolidation with air bronchograms is present. The findings are compatible with an acute infectious process. Impression: Pneumonia. Clinical correlation and follow-up chest radiograph after treatment are recommended.",
        "Right lower lobe airspace disease most likely pneumonia. There is no effusion or pneumothorax. Impression: Right lower lobe pneumonia.",
        "Right upper lobe airspace disease most likely pneumonia. There is no effusion or pneumothorax. Impression: Right upper lobe pneumonia."
    ],
    "Effusion": [
        "Pleural effusion is present bilaterally. Blunting of the costophrenic angles is observed. There is associated compressive atelectasis at the lung bases. Impression: Bilateral pleural effusions. Further clinical evaluation is advised.",
        "A moderate pleural effusion is identified on the left side. The left costophrenic angle is blunted. Impression: Moderate left pleural effusion.",
        "Small bilateral pleural effusions are noted. The costophrenic angles are blunted bilaterally. Impression: Small bilateral pleural effusions."
    ],
    "Cardiomegaly": [
        "The cardiac silhouette is enlarged, consistent with cardiomegaly. The cardiothoracic ratio is increased beyond normal limits. Mild pulmonary vascular congestion is noted. Impression: Cardiomegaly. Clinical correlation and echocardiography are recommended.",
        "Cardiomegaly is present with an increased cardiothoracic ratio. There is mild interstitial prominence suggesting early pulmonary edema. Impression: Cardiomegaly with possible early pulmonary edema.",
        "The heart is enlarged. Bilateral perihilar vascular congestion is noted. Impression: Cardiomegaly with pulmonary vascular congestion."
    ],
    "Atelectasis": [
        "Linear and subsegmental atelectasis is present at the lung bases bilaterally. No pneumothorax or pleural effusion is identified. Impression: Bibasilar atelectasis. Clinical correlation is recommended.",
        "Bibasilar atelectasis is noted. The cardiac silhouette is normal in size. Impression: Bibasilar atelectasis.",
        "Mild bibasilar atelectasis. No confluent lobar consolidation or pleural effusion. Impression: Bibasilar atelectasis."
    ],
    "Consolidation": [
        "Airspace consolidation is identified in the right lower lobe. Air bronchograms are present. Impression: Right lower lobe consolidation.",
        "There is lobar consolidation involving the left lower lobe. Air bronchograms are present. Impression: Left lower lobe consolidation.",
        "Bilateral airspace consolidation is noted. The findings are compatible with multifocal pneumonia. Impression: Multifocal consolidation."
    ],
    "Pneumothorax": [
        "A pneumothorax is present on the left side. The left lung is partially collapsed. The trachea is midline. Impression: Left pneumothorax. Urgent clinical evaluation is recommended.",
        "Right-sided pneumothorax is identified. There is a visible visceral pleural line with absence of lung markings. Impression: Right pneumothorax.",
        "Small to moderate sized right apical pneumothorax. No focal airspace consolidation is seen. Impression: Right apical pneumothorax."
    ],
    "Edema": [
        "Pulmonary edema is present with bilateral perihilar haziness. Increased interstitial markings and vascular congestion are noted. Impression: Pulmonary edema consistent with congestive heart failure.",
        "Bilateral interstitial pulmonary edema is identified. Bilateral pleural effusions are noted. Impression: Interstitial pulmonary edema.",
        "Pulmonary vascular congestion is present bilaterally. There is interstitial edema with peribronchial cuffing. Impression: Pulmonary edema."
    ],
    "Emphysema": [
        "Bilateral hyperinflation consistent with emphysema is present. The hemidiaphragms are flattened. Impression: Emphysematous changes consistent with chronic obstructive pulmonary disease.",
        "Emphysematous changes are present throughout both lungs. Bullous changes are noted in the upper lobes. Impression: Moderate to severe emphysema.",
        "Chronic obstructive changes consistent with emphysema are present. The lungs are hyperinflated with a low and flat diaphragm. Impression: Emphysema."
    ],
    "Fibrosis": [
        "Bilateral interstitial fibrosis is present. Reticular opacities are noted predominantly in the lower lobes. Impression: Pulmonary fibrosis. High resolution CT may be helpful.",
        "Interstitial lung disease with fibrotic changes is noted bilaterally. The lung volumes are reduced. Impression: Interstitial lung disease with fibrosis.",
        "Bilateral reticular opacities consistent with pulmonary fibrosis are present. Impression: Pulmonary fibrosis."
    ],
    "Infiltration": [
        "Bilateral pulmonary infiltrates are present. The infiltrates are predominantly interstitial in pattern. Impression: Bilateral interstitial infiltrates. Clinical correlation is strongly advised.",
        "Patchy infiltrates are identified in both lung fields. The infiltrates are more prominent in the lower lobes. Impression: Bilateral lower lobe infiltrates.",
        "Interstitial infiltrates are noted throughout the lungs bilaterally. Impression: Diffuse interstitial infiltrates."
    ],
    "Nodule": [
        "A solitary pulmonary nodule is identified in the right upper lobe. No spiculated margins are identified. Impression: Right upper lobe pulmonary nodule. Follow-up CT imaging is recommended.",
        "A small pulmonary nodule is present in the left mid lung. The nodule has smooth margins. Impression: Left mid lung nodule. Short-term follow-up imaging recommended.",
        "Multiple small pulmonary nodules are present bilaterally. Impression: Multiple bilateral pulmonary nodules. CT of the chest is recommended for further evaluation."
    ],
    "Mass": [
        "A large pulmonary mass is identified in the right upper lobe. The mass has irregular margins. Malignancy cannot be excluded. Impression: Right upper lobe mass, concerning for malignancy. Urgent CT recommended.",
        "A soft tissue mass is present in the left lung. The mass has lobulated margins. Impression: Left lung mass. Further evaluation with CT is recommended.",
        "A pulmonary mass is identified with associated pleural involvement. Impression: Pulmonary mass, concerning for malignancy."
    ],
    "Pleural_Thickening": [
        "Bilateral pleural thickening is noted. The pleural surfaces are irregular. Impression: Bilateral pleural thickening.",
        "Left-sided pleural thickening is present along the lateral chest wall. Impression: Left pleural thickening.",
        "Pleural thickening is noted at the lung bases bilaterally. Impression: Bilateral pleural thickening."
    ],
    "Hernia": [
        "A hiatal hernia is present with the gastric bubble noted above the diaphragm. The lungs are clear. Impression: Hiatal hernia.",
        "A large hiatal hernia is identified in the right lower thorax. Impression: Large hiatal hernia.",
        "Hiatal hernia is present with the stomach partially herniated into the thoracic cavity. Impression: Hiatal hernia."
    ],
}

# ============================================================
# IMAGE TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(DISEASES))
        )
    def forward(self, x):
        return self.head(self.gap(x.transpose(1, 2)).squeeze(-1))

# ============================================================
# LOAD MODELS (Simplified - No BioBART)
# ============================================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download model from Hugging Face if not exists
    model_path = "best_cls.pt"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model (349MB)... This may take 2-3 minutes"):
            response = requests.get(MODEL_URL, stream=True)
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    # Create Swin model
    swin = timm.create_model("swin_base_patch4_window7_224",
                             pretrained=False, num_classes=0, global_pool="")
    cls_head = ClassificationHead()
    
    # Load weights
    cls_ckpt = torch.load(model_path, map_location=device)
    swin.load_state_dict(cls_ckpt["swin"])
    cls_head.load_state_dict(cls_ckpt["cls_head"])
    
    swin = swin.to(device).eval()
    cls_head = cls_head.to(device).eval()
    
    return swin, cls_head, device

# ============================================================
# GET IMAGE FEATURES
# ============================================================
def get_image_features(swin, image_tensor):
    feats = swin.forward_features(image_tensor)
    if feats.dim() == 4:
        B, H, W, C = feats.shape
        feats = feats.view(B, H * W, C)
    return feats

# ============================================================
# GENERATE REPORT
# ============================================================
def generate_report(image, swin, cls_head, device, threshold=0.6):
    img_tensor = transform(image).unsqueeze(0).to(device)
    feats = get_image_features(swin, img_tensor)
    
    with torch.no_grad():
        logits = cls_head(feats)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Different thresholds for different diseases
    detected = []
    for i, disease in enumerate(DISEASES):
        if disease in ["Infiltration", "Nodule"]:
            if probs[i] > 0.65:
                detected.append((disease, float(probs[i])))
        elif probs[i] > threshold:
            detected.append((disease, float(probs[i])))
    
    detected.sort(key=lambda x: x[1], reverse=True)
    
    detected_names = [d[0] for d in detected]
    detected_probs = [d[1] for d in detected]
    
    if detected_names:
        top_disease = detected_names[0]
        report = random.choice(REPORT_TEMPLATES[top_disease])
    else:
        top_disease = "No Finding"
        report = random.choice(REPORT_TEMPLATES["No Finding"])
    
    return detected_names, detected_probs, report, top_disease

# ============================================================
# MAIN APP
# ============================================================
with st.spinner("Loading models..."):
    swin, cls_head, device = load_models()
st.success("✅ Models ready")

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("Upload X-Ray")
    uploaded = st.file_uploader("Choose a chest X-ray image",
                                type=["jpg", "png", "jpeg"],
                                label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with col_right:
    st.subheader("Report")

    if uploaded:
        if st.button("Generate Report", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                detected_names, detected_probs, report, top_disease = \
                    generate_report(image, swin, cls_head, device, threshold=0.6)

            st.markdown("### Findings")
            if not detected_names:
                st.success("✅ Normal — No significant findings detected")
            else:
                df = pd.DataFrame({
                    "Finding": detected_names,
                    "Confidence": [f"{p:.1%}" for p in detected_probs]
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Radiology Report")
            if top_disease != "No Finding":
                st.caption(f"Primary finding: **{top_disease}** ({detected_probs[0]:.1%})")
            st.write(report)

            st.download_button(
                "📥 Download Report",
                data=f"PRIMARY FINDING: {top_disease}\n\nALL FINDINGS:\n" + 
                     (f"None (Normal)" if not detected_names else 
                      "\n".join([f"{n}: {p:.1%}" for n, p in zip(detected_names, detected_probs)])) +
                     f"\n\nREPORT:\n{report}",
                file_name="xray_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Upload an image to begin")

st.markdown("---")
st.caption("⚠️ This is an AI FYP Prototype. Please consult a real medical professional after viewing the report.")