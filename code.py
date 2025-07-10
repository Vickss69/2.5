import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time
import traceback

st.set_page_config(page_title="Beauty Score Comparator", layout="wide")

st.info("📝 NOTE: This analysis is based on geometric features in the image, not real-life standards of beauty.")
st.warning("⚠️ DISCLAIMER: Use this application responsibly and for entertainment purposes only.")

# --- OpenCV Model Initialization (Cached for Performance) ---
@st.cache_resource
def load_opencv_models():
    """Loads and caches the OpenCV Haar Cascade models."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

# Preload models
face_cascade, eye_cascade = load_opencv_models()

# --- Analysis Functions (Rewritten for OpenCV) ---

def analyze_image(image_path, progress_callback=None):
    """
    Main analysis function using only OpenCV.
    Detects face and eyes, then computes all scores.
    """
    try:
        if progress_callback: progress_callback(0, "Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            return {k: 0 for k in ['face_shape', 'skin_score', 'eye_score', 'hair_score', 'final_score']}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Face Detection
        if progress_callback: progress_callback(15, "Detecting face...")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            if progress_callback: progress_callback(100, "No face detected!")
            # Still try to rate hair if no face is found
            hair_score = rate_hair(image)
            return {'face_shape': 0, 'skin_score': 0, 'eye_score': 0, 'hair_score': hair_score, 'final_score': hair_score * 0.15}
            
        # Use the largest detected face
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_roi = image[y:y+h, x:x+w]
        
        # 2. Face Shape Score
        if progress_callback: progress_callback(30, "Analyzing face shape...")
        # Aspect ratio of the bounding box
        aspect_ratio = h / w if w > 0 else 0
        # Ideal is oval-like, around 1.3-1.5
        face_shape_score = max(0, 100 - abs(aspect_ratio - 1.4) * 100)
        
        # 3. Skin Score
        if progress_callback: progress_callback(50, "Analyzing skin clarity...")
        # Score based on brightness and clarity (low standard deviation)
        brightness = np.mean(face_roi)
        clarity = np.std(face_roi)
        skin_score = np.clip((brightness/2.5) + (50-clarity), 0, 100)
        
        # 4. Eye Score (Shape + Color)
        if progress_callback: progress_callback(70, "Analyzing eyes...")
        eye_score = rate_eyes(face_roi, gray[y:y+h, x:x+w])

        # 5. Hair Score
        if progress_callback: progress_callback(85, "Analyzing hair...")
        hair_score = rate_hair(image)
        
        # 6. Final Weighted Score
        final_score = (
            face_shape_score * 0.35 +
            skin_score * 0.35 +
            eye_score * 0.15 +
            hair_score * 0.15
        )

        if progress_callback: progress_callback(100, "Analysis complete!")
        return {'face_shape': face_shape_score, 'skin_score': skin_score, 'eye_score': eye_score, 'hair_score': hair_score, 'final_score': final_score}

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return {k: 0 for k in ['face_shape', 'skin_score', 'eye_score', 'hair_score', 'final_score']}

def rate_eyes(face_roi, face_gray):
    """Detects eyes within the face ROI and scores them."""
    eyes = eye_cascade.detectMultiScale(face_gray)
    if len(eyes) < 2:
        return 0
    
    # Use the two largest detected eyes
    eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
    
    eye_scores = []
    for (ex, ey, ew, eh) in eyes:
        # Shape Score
        eye_ar = eh / ew if ew > 0 else 0
        shape_score = max(0, 100 - abs(eye_ar - 0.45) * 200) # Idealize ~0.45 AR

        # Color score (simpler version)
        eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
        b, g, r = np.mean(eye_roi, axis=(0, 1))
        
        if b > 120 and b > r and b > g: color_score = 95  # Blue/Gray
        elif g > 90 and g > r: color_score = 85  # Green/Hazel
        else: color_score = 75 # Brown
        
        eye_scores.append(shape_score * 0.6 + color_score * 0.4)
        
    return np.mean(eye_scores) if eye_scores else 0

def rate_hair(image):
    """Simple heuristic for hair color and texture."""
    try:
        height, width, _ = image.shape
        hair_region = image[:int(height * 0.25), :]
        if hair_region.size == 0: return 0

        gray_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray_hair, cv2.CV_64F).var()
        texture_score = np.clip(texture, 0, 100)
        return texture_score
    except Exception:
        return 0

def mark_winner(image_path):
    """Draws a 'WINNER' banner on the image."""
    image = Image.open(image_path).convert("RGBA")
    txt_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    text = "WINNER"
    font_size = int(image.height / 10)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError: # Fallback for older PIL
        text_width, text_height = draw.textsize(text, font=font)
    
    pos_x = (image.width - text_width) / 2
    pos_y = image.height - text_height * 1.5
    
    # Golden banner
    draw.rectangle(
        (0, pos_y - 5, image.width, pos_y + text_height + 5),
        fill=(255, 215, 0, 180) # Semi-transparent gold
    )
    draw.text((pos_x, pos_y), text, fill="white", font=font, stroke_width=2, stroke_fill="black")
    return Image.alpha_composite(image, txt_layer)

# --- Streamlit UI (No changes needed here) ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")

with col2:
    st.subheader("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    temp_file1_path, temp_file2_path = None, None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file1:
            temp_file1.write(uploaded_file1.getvalue())
            temp_file1_path = temp_file1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file2:
            temp_file2.write(uploaded_file2.getvalue())
            temp_file2_path = temp_file2.name

        def update_progress(percent, message=""):
            progress_bar.progress(int(percent))
            if message: status_text.text(message)
        
        metrics1 = analyze_image(temp_file1_path, lambda p, m: update_progress(p / 2, f"Image 1: {m}"))
        metrics2 = analyze_image(temp_file2_path, lambda p, m: update_progress(50 + p / 2, f"Image 2: {m}"))

        status_text.text("Comparison complete!")
        time.sleep(1)
        status_text.empty(); progress_bar.empty()
        
        s1, s2 = metrics1['final_score'], metrics2['final_score']

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(uploaded_file1, caption=f"Image 1 | Score: {s1:.2f}")
            with st.expander("Detailed Scores for Image 1"):
                st.dataframe(metrics1)

        with res_col2:
            st.image(uploaded_file2, caption=f"Image 2 | Score: {s2:.2f}")
            with st.expander("Detailed Scores for Image 2"):
                st.dataframe(metrics2)

        st.markdown("---")
        st.subheader("🏆 The Winner Is...")

        if s1 == 0 and s2 == 0:
            st.error("Could not evaluate either image properly.")
        elif s1 >= s2:
            st.image(mark_winner(temp_file1_path), caption="Image 1 Wins!", use_column_width=True)
        else:
            st.image(mark_winner(temp_file2_path), caption="Image 2 Wins!", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())
    
    finally:
        if temp_file1_path and os.path.exists(temp_file1_path): os.unlink(temp_file1_path)
        if temp_file2_path and os.path.exists(temp_file2_path): os.unlink(temp_file2_path)
else:
    st.info("Upload two images to begin.")
