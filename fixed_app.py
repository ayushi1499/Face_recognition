
# Fixed and unified version of VisionText AI Hub
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os
import av
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
from transformers import pipeline
from deepface import DeepFace
import tempfile
import tensorflow as tf
import base64
import json
import requests
import time

# Page configuration
st.set_page_config(page_title="VisionText AI Hub", page_icon="🧠", layout="wide")

# Custom styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #4285F4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.8rem;
    color: #0f9d58;
    margin-bottom: 1rem;
}
.section {
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>VisionText AI Hub</h1>", unsafe_allow_html=True)
st.markdown("""This application combines face recognition, face analysis, and text summarization features.""")

# Setup
def create_directories():
    dataset_path = "./face_dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    return dataset_path

create_directories()

if "summarizer" not in st.session_state:
    st.session_state.summarizer = pipeline("summarization")
if "face_cascade" not in st.session_state:
    st.session_state.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Helper functions
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Tabs
tabs = st.tabs(["Home", "Face Recognition", "Face Analysis", "Text Summarization"])

with tabs[0]:
    st.markdown("<h2 class='sub-header'>Welcome to the VisionText AI Hub!</h2>", unsafe_allow_html=True)
    st.markdown("""
    This app offers:
    - Face Recognition
    - Face Analysis (emotion, age, etc.)
    - Text Summarization
    """)
    col1, col2, col3 = st.columns(3)
    with col1: st.image("https://cdn.prod.website-files.com/66cd2cbca244787c4d744cc5/67362473a7b512c05960969b_OMY%20(1).jpeg", caption="Face Recognition")
    with col2: st.image("https://static.vecteezy.com/system/resources/previews/002/006/186/original/face-recognition-icon-illustration-vector.jpg", caption="Face Analysis")
    with col3: st.image("https://miro.medium.com/v2/resize:fit:1064/1*GIVviyN9Q0cqObcy-q-juQ.png", caption="Text Summarization")

with tabs[1]:
    st.subheader("Face Recognition")
    mode = st.radio("Choose an operation", ["Register New Face", "Recognize Faces"])

    class FaceRegisterProcessor(VideoProcessorBase):
        def __init__(self):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_data = []
            self.counter = 0
            self.max_samples = 30
            self.name = ""
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if self.name and len(self.face_data) < self.max_samples and len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                face_section = img[y:y+h, x:x+w]
                face_section = cv2.resize(face_section, (100, 100))
                self.counter += 1
                if self.counter % 5 == 0:
                    self.face_data.append(face_section)
                cv2.putText(img, f"Samples: {len(self.face_data)}/{self.max_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        def set_name(self, name): self.name = name
        def get_face_data(self): return np.asarray(self.face_data) if self.face_data else None

    if mode == "Register New Face":
        name = st.text_input("Enter name:")
        ctx = webrtc_streamer(key="register", video_processor_factory=FaceRegisterProcessor)
        if ctx.video_processor and name:
            ctx.video_processor.set_name(name)
            if st.button("Save Face Data"):
                data = ctx.video_processor.get_face_data()
                if data is not None:
                    data = data.reshape((data.shape[0], -1))
                    np.save(os.path.join(create_directories(), name + ".npy"), data)
                    st.success(f"{len(data)} samples saved for {name}.")
                else:
                    st.error("No data collected.")

with tabs[2]:
    st.subheader("Face Analysis")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file.name)
            analysis = DeepFace.analyze(img_path=tmp_file.name, actions=["age", "gender", "emotion", "race"])
            st.json(analysis[0])

with tabs[3]:
    st.subheader("Text Summarization")
    input_text = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if input_text:
            summary = st.session_state.summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            st.success(summary[0]['summary_text'])
        else:
            st.warning("Please enter some text.")

