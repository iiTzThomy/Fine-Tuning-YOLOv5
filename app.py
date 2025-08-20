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
font = ImageFont.truetype("arialbd.ttf", 25)
pathlib.PosixPath = pathlib.WindowsPath

sys.path.append(os.path.abspath("C:/Users/a944352/Documents/Fine tuning Yolo/yolov5"))

from models.experimental import attempt_load
from utils.general import non_max_suppression

def load_model(model_path):
    model = attempt_load(model_path, device=torch.device('cpu'))
    model.eval()
    return model

model = load_model("C:/Users/a944352/Documents/Fine tuning Yolo/yolov5/runs/train/yolov5_accidents3/weights/best.pt")

st.title("Détection d'accidents avec YOLOv5")

mode = st.radio("Choisir un mode :", ["Image", "Vidéo"])

# --- MODE IMAGE ---
if mode == "Image":
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        # Prétraitement
        img_resized = image.resize((640, 640))
        img = np.array(img_resized)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        
        pred = model(img)[0]
        pred = non_max_suppression(pred)[0]

        # Couleurs par classe
        class_colors = {i: tuple(random.choices(range(256), k=3)) for i, _ in enumerate(model.names)}

        img_draw = img_resized.copy()
        draw = ImageDraw.Draw(img_draw)
        details = []

        if pred is not None and len(pred):
            for *box, conf, cls in pred:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                color = class_colors[int(cls)]
                color2 = tuple(int(c) for c in color)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                draw.text((x1, y1), f"{model.names[int(cls)]} {conf:.2f}", fill="lime", font=font)
                details.append(f"Classe: {model.names[int(cls)]}, confiance: {conf.item():.2f}")

            st.image(img_draw, caption="Détections", use_container_width=True)
            st.subheader("Détails de la détection")
            for d in details:
                st.write(d)
        else:
            st.image(img_draw, caption="Détections", use_container_width=True)
            st.write("Aucune détection.")

# --- MODE VIDÉO ---
elif mode == "Vidéo":
    uploaded_video = st.file_uploader("Choisir une vidéo", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))

        stframe = st.empty()
        st.info("Traitement de la vidéo en cours...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Prétraitement (resize 640x640)
            img_resized = cv2.resize(frame, (640, 640))
            img = img_resized[:, :, ::-1]  # BGR -> RGB
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).float()

            # Inférence
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

            if pred is not None and len(pred):
                for *box, conf, cls in pred:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_resized, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Ajouter la frame annotée
            stframe.image(img_resized, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Vidéo traitée !")
