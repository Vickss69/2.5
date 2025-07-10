import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time
import mediapipe as mp
import traceback
import math

st.set_page_config(page_title="Beauty Score Comparator", layout="wide")

st.info("📝 NOTE: This analysis is based on geometric and colorimetric features in the image, not real-life standards of beauty.")
st.warning("⚠️ DISCLAIMER: Use this application responsibly and for entertainment purposes only.")

# --- MediaPipe Initialization (Cached for Performance) ---
@st.cache_resource
def load_mediapipe_models():
    """Loads and caches the MediaPipe Face Mesh model."""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
    return face_mesh

# Preload the model
face_mesh_model = load_mediapipe_models()

# --- Landmark Definitions (MediaPipe Indices) ---
# These are key points on the 468-landmark mesh.
# You can find maps online: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
JAWLINE_INDICES = [61, 291, 199, 175, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# --- Analysis Functions (Refactored for MediaPipe) ---

def get_landmarks(image_path, face_mesh):
    """Detects face landmarks using MediaPipe and returns them in pixel coordinates."""
    try:
        image = cv2.imread(image_path)
        if image is None: return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        h, w, _ = image.shape
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks_px = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.int32)
        return landmarks_px, image
    except Exception:
        return None

def calculate_face_shape(landmarks):
    """Calculates face shape score from MediaPipe landmarks."""
    # Using more stable landmarks for dimensions
    face_width = np.linalg.norm(landmarks[234] - landmarks[454]) # Left to right cheek
    face_height = np.linalg.norm(landmarks[10] - landmarks[152]) # Forehead top to chin tip
    jaw_width = np.linalg.norm(landmarks[58] - landmarks[288])
    
    if face_width == 0 or jaw_width == 0: return 0

    aspect_ratio = face_height / face_width
    jaw_to_face_ratio = jaw_width / face_width
    
    # Simple classification, can be expanded.
    if aspect_ratio > 1.05 and 0.9 < jaw_to_face_ratio < 1.0: score = 95 # Oval
    elif 0.95 <= aspect_ratio <= 1.05 and jaw_to_face_ratio > 0.95: score = 80 # Round/Square
    elif aspect_ratio > 1.0 and jaw_to_face_ratio < 0.9: score = 88 # Heart
    else: score = 75 # Other
    
    return score

def get_skin_score(landmarks, image):
    """Calculates average skin color score from a tight bounding box around landmarks."""
    try:
        (x_min, y_min) = np.min(landmarks, axis=0)
        (x_max, y_max) = np.max(landmarks, axis=0)
        
        face_roi = image[y_min:y_max, x_min:x_max]
        if face_roi.size == 0: return 0
        
        avg_bgr = np.mean(face_roi, axis=(0, 1))
        # Brightness/Clarity Score
        brightness = np.linalg.norm(avg_bgr)
        max_brightness = np.sqrt(3 * 255**2)
        clarity_score = (brightness / max_brightness) * 100

        # Uniformity score (lower std deviation is better)
        std_dev = np.mean(np.std(face_roi, axis=(0, 1)))
        uniformity_score = max(0, 100 - (std_dev * 2)) # Heuristic

        return (clarity_score * 0.6 + uniformity_score * 0.4)
    except Exception:
        return 0

def rate_jawline(landmarks):
    """Rates jawline sharpness and symmetry using specific jawline indices."""
    jawline_points = landmarks[JAWLINE_INDICES]
    chin_point = landmarks[152]

    # Symmetry
    left_jaw = jawline_points[:8]
    right_jaw = jawline_points[9:][::-1] # Reverse for comparison
    left_dists = np.linalg.norm(left_jaw - chin_point, axis=1)
    right_dists = np.linalg.norm(right_jaw - chin_point, axis=1)
    symmetry_diff = np.mean(np.abs(left_dists - right_dists))
    symmetry_score = max(0, 100 - symmetry_diff)

    # Sharpness (angle)
    angles = []
    for i in range(1, len(jawline_points) - 1):
        p1, p2, p3 = jawline_points[i-1], jawline_points[i], jawline_points[i+1]
        ba = p1 - p2; bc = p3 - p2
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
        angles.append(angle)
    
    sharpness_deviation = np.mean(np.abs(np.array(angles) - 150)) # Idealize a wider, smoother angle
    sharpness_score = max(0, 100 - sharpness_deviation * 2)

    return 0.6 * symmetry_score + 0.4 * sharpness_score

def rate_eyes(landmarks, image):
    """Calculates scores for eye shape and color."""
    try:
        # Eye Shape (Aspect Ratio)
        left_eye_pts = landmarks[LEFT_EYE_INDICES]
        right_eye_pts = landmarks[RIGHT_EYE_INDICES]
        
        # Left eye
        left_w = np.linalg.norm(left_eye_pts[0] - left_eye_pts[8])
        left_h = np.linalg.norm(left_eye_pts[4] - left_eye_pts[12])
        left_ar = left_h / left_w if left_w != 0 else 0
        # Right eye
        right_w = np.linalg.norm(right_eye_pts[0] - right_eye_pts[8])
        right_h = np.linalg.norm(right_eye_pts[4] - right_eye_pts[12])
        right_ar = right_h / right_w if right_w != 0 else 0
        
        avg_ar = (left_ar + right_ar) / 2
        # Score based on 'almond-eye' ideal (~0.3-0.4 AR)
        shape_score = max(0, 100 - abs(avg_ar - 0.35) * 300)

        # Eye Color
        (lx_min, ly_min), (lx_max, ly_max) = np.min(left_eye_pts, axis=0), np.max(left_eye_pts, axis=0)
        (rx_min, ry_min), (rx_max, ry_max) = np.min(right_eye_pts, axis=0), np.max(right_eye_pts, axis=0)
        
        left_eye_roi = image[ly_min:ly_max, lx_min:lx_max]
        right_eye_roi = image[ry_min:ry_max, rx_min:rx_max]

        if left_eye_roi.size == 0 or right_eye_roi.size == 0: return shape_score, 0
        
        avg_color_bgr = (np.mean(left_eye_roi, axis=(0,1)) + np.mean(right_eye_roi, axis=(0,1))) / 2
        b, g, r = avg_color_bgr
        
        # Scoring rare colors higher
        if b > r and b > g and b > 110: color_score = 95 # Blue/Gray
        elif g > r and g > b and g > 90: color_score = 85 # Green
        elif abs(r - g) < 20 and r > 100: color_score = 80 # Hazel/Amber
        else: color_score = 70 # Brown
        
        return shape_score, color_score
    except Exception:
        return 0, 0

def rate_hair(image):
    """Simple heuristic for hair color and texture."""
    try:
        # Define hair region as top 25% of the image
        height, width, _ = image.shape
        hair_region = image[:int(height * 0.25), :]
        if hair_region.size == 0: return 0

        # Color score (less common colors are often perceived as more striking)
        avg_bgr = np.mean(hair_region, axis=(0, 1))
        # Simple score based on darkness/lightness
        brightness = np.linalg.norm(avg_bgr)
        color_score = 50 + ((brightness - 128) / 2.5) # Center around 50

        # Texture/Density score via edge detection
        gray_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_hair, 100, 200)
        density = np.sum(edges > 0) / edges.size
        density_score = min(100, density * 500) # Amplify score
        
        return np.clip((color_score * 0.4 + density_score * 0.6), 0, 100)
    except Exception:
        return 0

def analyze_image(image_path, face_mesh_model, progress_callback=None):
    """Main analysis function that calls all sub-modules."""
    metrics = {k: 0 for k in ['face_shape', 'skin_score', 'jawline', 'eye_shape', 'eye_color', 'hair_score', 'final_score']}
    
    if progress_callback: progress_callback(0, "Detecting face landmarks...")
    
    landmark_data = get_landmarks(image_path, face_mesh_model)
    if landmark_data is None:
        if progress_callback: progress_callback(100, "No face detected!")
        return metrics
    
    landmarks, image = landmark_data
    
    if progress_callback: progress_callback(20, "Analyzing face shape...")
    metrics['face_shape'] = calculate_face_shape(landmarks)
    
    if progress_callback: progress_callback(35, "Analyzing skin clarity...")
    metrics['skin_score'] = get_skin_score(landmarks, image)
    
    if progress_callback: progress_callback(50, "Analyzing jawline...")
    metrics['jawline'] = rate_jawline(landmarks)
    
    if progress_callback: progress_callback(65, "Analyzing eyes...")
    metrics['eye_shape'], metrics['eye_color'] = rate_eyes(landmarks, image)

    if progress_callback: progress_callback(80, "Analyzing hair...")
    metrics['hair_score'] = rate_hair(image)
    
    # Calculate weighted final score
    metrics['final_score'] = (
        metrics['face_shape'] * 0.25 + 
        metrics['skin_score'] * 0.25 + 
        metrics['jawline']    * 0.20 + 
        metrics['eye_shape']  * 0.15 + 
        metrics['eye_color']  * 0.10 + 
        metrics['hair_score'] * 0.05
    )
    
    if progress_callback: progress_callback(100, "Analysis complete!")
    return metrics

def mark_winner(image_path):
    # Same as before, but with better font handling
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    text = "WINNER"
    font_path = "arial.ttf" # A common font
    font_size = int(image.height / 10)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        st.warning("Arial font not found. Using default font.")

    try: bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError: text_width, text_height = draw.textsize(text, font=font)
    else: text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pos_x = (image.width - text_width) / 2
    pos_y = image.height - text_height * 1.5
    draw.text((pos_x, pos_y), text, fill=(255, 215, 0), font=font, stroke_width=2, stroke_fill="black")
    return image

# --- Streamlit UI ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")

with col2:
    st.subheader("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not a None:
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
        
        metrics1 = analyze_image(temp_file1_path, face_mesh_model, lambda p, m: update_progress(p / 2, f"Image 1: {m}"))
        metrics2 = analyze_image(temp_file2_path, face_mesh_model, lambda p, m: update_progress(50 + p / 2, f"Image 2: {m}"))

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
            st.error("Could not detect faces in either image.")
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
