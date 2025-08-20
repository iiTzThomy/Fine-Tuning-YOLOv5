import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys
import pathlib
import random
import cv2
import tempfile
import time
from io import BytesIO

# ================================
# CONFIG & SETUP
# ================================
st.set_page_config(page_title="Accident Detection YOLOv5", layout="wide")
# Police : fallback si arialbd.ttf n'est pas présent
try:
    font = ImageFont.truetype("arialbd.ttf", 25)
except:
    font = ImageFont.load_default()

# Hack WindowsPath si nécessaire
pathlib.PosixPath = pathlib.WindowsPath

# >>>> ADAPTE CE CHEMIN À TON INSTALL <<<<
YOLOV5_DIR = "C:/Users/a944352/Documents/Fine tuning Yolo/yolov5"
MODEL_PATH = "C:/Users/a944352/Documents/Fine tuning Yolo/yolov5/runs/train/yolov5_accidents3/weights/best.pt"

sys.path.append(os.path.abspath(YOLOV5_DIR))

# Import YOLOv5 utils
from models.experimental import attempt_load
from utils.general import non_max_suppression

# ================================
# charger YOLO MODEL
# ================================
@st.cache_resource
def load_model(model_path):
    model = attempt_load(model_path, device=torch.device('cpu'))
    model.eval()
    return model

model = load_model(MODEL_PATH)

# ================================
# SIDEBAR
# ================================
st.sidebar.title("⚙️ Paramètres")
mode = st.sidebar.radio("Mode", ["Image", "Vidéo"], index=0)
conf_thres = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.01)
iou_thres = st.sidebar.slider("Seuil IoU (NMS)", 0.0, 1.0, 0.45, 0.01)
if mode == "Image":
    img_size = st.sidebar.selectbox("Taille de l'image", [320, 416, 640], index=2)
show_labels = st.sidebar.checkbox("Afficher les labels", value=True)
show_scores = st.sidebar.checkbox("Afficher les scores", value=True)

tracking_enabled = False
if mode == "Vidéo":
    tracking_enabled = st.sidebar.checkbox("Activer le tracking (DeepSORT)", value=False)

# ================================
# Couleurs des classes
# ================================
class_colors = {i: tuple(random.choices(range(256), k=3)) for i, _ in enumerate(model.names)}

# ================================
# HELPERS
# ================================
def preprocess_pil(image: Image.Image, size: int):
    """Resize + to tensor [1,3,H,W] in float32 normalized 0..1"""
    img_resized = image.resize((size, size))
    img = np.array(img_resized) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img_resized, torch.from_numpy(img)

def preprocess_cv(frame: np.ndarray, size: int):
    """Resize + to tensor from BGR frame"""
    img_resized = cv2.resize(frame, (size, size))
    img = img_resized[:, :, ::-1] / 255.0  # BGR -> RGB
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img_resized, torch.from_numpy(img)

def draw_box_pil(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, color, label_text=None):
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    if label_text:
        draw.text((x1, max(0, y1 - 24)), label_text, fill=(0,0,0), font=font)

def pil_to_bytes(img: Image.Image, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ================================
# MODE IMAGE
# ================================
if mode == "Image":
    st.markdown("<h2>📷 Détection d'accidents — Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([2, 1])

    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            img_resized, tensor = preprocess_pil(image, img_size)

            # Inférence
            with torch.no_grad():
                pred = model(tensor)[0]
                pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]

            img_draw = img_resized.copy()
            draw = ImageDraw.Draw(img_draw)
            details = []
            per_class_counts = {model.names[i]: 0 for i in range(len(model.names))}

            if pred is not None and len(pred):
                for *box, conf, cls in pred:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    color = class_colors[int(cls)]
                    label_text = None
                    if show_labels and show_scores:
                        label_text = f"{model.names[int(cls)]} {conf:.2f}"
                    elif show_labels:
                        label_text = f"{model.names[int(cls)]}"
                    draw_box_pil(draw, x1, y1, x2, y2, color, label_text)
                    per_class_counts[model.names[int(cls)]] += 1
                    details.append(f"• {model.names[int(cls)]} — conf: {conf.item():.2f} — box: ({x1},{y1})→({x2},{y2})")

            st.image(img_draw, caption="Résultat", use_container_width=True)

            # download image annotated
            dl_buf = pil_to_bytes(img_draw, "PNG")
            st.download_button(
                "📥 Télécharger l'image annotée (PNG)",
                data=dl_buf,
                file_name="result_annotated.png",
                mime="image/png"
            )
        else:
            st.info("Charge une image pour lancer la détection.")

    with col2:
        if uploaded_file is not None:
            st.markdown("### 📊 Statistiques")
            total = sum(per_class_counts.values())
            st.write(f"**Total détections :** {total}")
            for cname, cnt in per_class_counts.items():
                st.write(f"- {cname}: **{cnt}**")

            st.markdown("### 📝 Détails")
            if len(details) == 0:
                st.write("Aucune détection.")
            else:
                for d in details:
                    st.write(d)

# ================================
# MODE VIDÉO
# ================================
elif mode == "Vidéo":
    format_vid = st.sidebar.selectbox("Format de la vidéo", [320, 480, 640], index=2)
    st.markdown("<h2>🎥 Détection d'accidents — Vidéo</h2>", unsafe_allow_html=True)
    uploaded_video = st.sidebar.file_uploader("Choisis une vidéo", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # lire la vidéo uploadée dans un fichier temporaire
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_in.write(uploaded_video.read())
        t_in.flush()
        cap = cv2.VideoCapture(t_in.name)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # fichier de sortie annoté
        t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(t_out.name, fourcc, fps, (format_vid, format_vid))

        # UI live
        stframe = st.empty()
        progress = st.progress(0)
        info_col, stats_col = st.columns([2, 1])

        # Tracking (optionnel)
        tracker = None
        if tracking_enabled:
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
                tracking_ids_seen = set()
            except Exception as e:
                st.error("DeepSORT indisponible. Installe : `pip install deep-sort-realtime`")
                tracking_enabled = False

        # stats
        stframe = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames = 0
        t0 = time.time()
        detections_total = 0
        per_class_counts = {model.names[i]: 0 for i in range(len(model.names))}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1

            # Prétraitement
            img_resized, tensor = preprocess_cv(frame, format_vid)

            # Inférence YOLO
            with torch.no_grad():
                pred = model(tensor)[0]
                pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]

            # Dessin
            if tracking_enabled and tracker is not None:
                # Préparer detections : (xywh), conf, class_id
                dets = []
                if pred is not None and len(pred):
                    for *box, conf, cls in pred:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        w, h = x2 - x1, y2 - y1
                        dets.append(((x1, y1, w, h), float(conf.item()), int(cls)))

                        # stats par détection (compte par frame)
                        detections_total += 1
                        per_class_counts[model.names[int(cls)]] += 1

                tracks = tracker.update_tracks(dets, frame=img_resized)
                for trk in tracks:
                    if not trk.is_confirmed():
                        continue
                    l, t, r, b = map(int, trk.to_ltrb())
                    tid = trk.track_id
                    tracking_ids_seen.add(tid)
                    # Nom de classe si disponible dans association
                    label_txt = f"ID:{tid}"
                    cv2.rectangle(img_resized, (l, t), (r, b), (0, 255, 0), 2)
                    if show_labels:
                        cv2.putText(img_resized, label_txt, (l, max(10, t - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                # Sans tracking : dessiner simplement les boxes
                if pred is not None and len(pred):
                    for *box, conf, cls in pred:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        color = class_colors[int(cls)]
                        label_txt = None
                        if show_labels and show_scores:
                            label_txt = f"{model.names[int(cls)]} {conf:.2f}"
                        elif show_labels:
                            label_txt = f"{model.names[int(cls)]}"
                        cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
                        if label_txt:
                            cv2.putText(img_resized, label_txt, (x1, max(10, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                        # stats par détection
                        detections_total += 1
                        per_class_counts[model.names[int(cls)]] += 1

            # Ecrire frame annotée et afficher
            out_writer.write(img_resized)
            stframe.image(img_resized, channels="BGR", use_container_width=True)

            stats_placeholder.markdown(f"""**Frame** : {frames}/{total_frames}**
                                    Détections totales** : {len(pred) if pred is not None else 0}""")
            progress_bar.progress(int(frames / total_frames * 100) if total_frames > 0 else 1.0)
            # Progress
            if total_frames > 0:
                progress.progress(min(1.0, frames / total_frames))
            else:
                # fallback si frame count inconnu
                if frames % 10 == 0:
                    progress.progress(min(1.0, (frames % 100) / 100.0))

            # stats side
            with stats_col:
                elapsed = max(1e-6, time.time() - t0)
                fps_est = frames / elapsed
                st.markdown("### 📊 Stats")
                st.write(f"Frames traitées : **{frames}**")
                st.write(f"FPS (approx) : **{fps_est:.1f}**")
                st.write(f"Détections totales : **{detections_total}**")
                for cname, cnt in per_class_counts.items():
                    st.write(f"- {cname}: **{cnt}**")
                if tracking_enabled and 'tracking_ids_seen' in locals():
                    st.write(f"Objets suivis uniques : **{len(tracking_ids_seen)}**")

        cap.release()
        out_writer.release()

        # Téléchargement de la vidéo annotée
        with open(t_out.name, "rb") as f:
            st.download_button(
                "📥 Télécharger la vidéo annotée (MP4)",
                data=f.read(),
                file_name="video_annotated.mp4",
                mime="video/mp4"
            )

        st.success("✅ Vidéo terminée !")
