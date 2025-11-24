import os
# --- DOTENV IMPORT ---
from dotenv import load_dotenv
# Load environment variables from a .env file (must be in the same directory)
load_dotenv()
# --- END DOTENV IMPORT ---

import cv2
import numpy as np
import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
import time

# --- Configuration ---
st.set_page_config(
    page_title="🪪 Myanmar Driving License Extractor (AI OCR)",
    layout="wide"
)

# --- 1. Core AI OCR Engine Setup ---
from google import genai
from google.genai import types
from PIL import Image

# Initialize the AI Client
try:
    # The client automatically uses the GEMINI_API_KEY loaded by load_dotenv()
    client = genai.Client()
except Exception as e:
    st.error(f"Error initializing AI client. Please ensure your GEMINI_API_KEY is set correctly in your .env file. Details: {e}")
    st.stop()


# --- 2. Data Extraction Prompt and Schema ---

# Define the expected output structure using Pydantic for robust parsing
extraction_schema = {
    "type": "object",
    "properties": {
        "license_no": {"type": "string", "description": "The driving license number, typically like 'A/123456/22'."},
        "name": {"type": "string", "description": "The full name of the license holder."},
        "nrc_no": {"type": "string", "description": "The NRC ID number, typically like '12/MASANA(N)123456'."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "blood_type": {"type": "string", "description": "The blood type, e.g., 'A+', 'B', 'O-', 'AB'."},
        "valid_up": {"type": "string", "description": "The license expiry date in DD-MM-YYYY format."}
    },
    "required": ["license_no", "name", "nrc_no", "date_of_birth", "blood_type", "valid_up"]
}

# The main prompt for the AI model
EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Driving License.
Extract the following key data fields and return the result strictly as a JSON object matching the provided schema: 
License No, Name, NRC No, Date of Birth, Blood Type, and Valid Up (Expiry Date).
If a field is not found, return an empty string "" for that value.
Do not include any extra text or formatting outside of the JSON object.
Ensure the extracted dates are in the DD-MM-YYYY format.
"""

# --- 3. Image Processing Functions (Simplified) ---

def handle_file_to_pil(uploaded_file):
    """Converts uploaded file or bytes to a PIL Image object."""
    if uploaded_file is None:
        return None
        
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    try:
        # Use PIL to open directly from bytes
        image_pil = Image.open(BytesIO(file_bytes))
        return image_pil
    except Exception as e:
        st.error(f"Error converting file to PIL image: {e}")
        return None
        
def handle_pil_to_cv2(image_pil):
    """Converts PIL Image to a CV2 image object (BGR)."""
    if image_pil is None:
        return None
    # Convert PIL to NumPy array (RGB)
    img_np_rgb = np.array(image_pil.convert('RGB'))
    # Convert RGB to BGR for OpenCV
    img_cv_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    return img_cv_bgr

# The preprocessing function is removed, as it's no longer needed.

# --- 4. AI OCR and Extraction Logic ---

def ai_extract_data_structured(image_pil):
    """
    Uses the AI API to analyze the image and extract structured data.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[EXTRACTION_PROMPT, image_pil],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=0.0, # Use low temperature for deterministic data extraction
            )
        )
        
        # The response.text is a JSON string matching the schema
        import json
        structured_data = json.loads(response.text)
        return structured_data
        
    except genai.errors.APIError as e:
        st.error(f"AI API Error: Could not process the image. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

# --- 5. Helper Functions ---

def create_downloadable_files(extracted_dict):
    """Formats the extracted data into CSV, TXT, and DOC formats."""
    # Convert the flat dictionary to the desired output format for display/download
    results_dict = {
        "License No": extracted_dict.get('license_no', ''),
        "Name": extracted_dict.get('name', ''),
        "NRC No": extracted_dict.get('nrc_no', ''),
        "Date of Birth": extracted_dict.get('date_of_birth', ''),
        "Blood Type": extracted_dict.get('blood_type', ''),
        "Valid Up": extracted_dict.get('valid_up', '')
    }
    
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    
    return txt_content, csv_content, doc_content, results_dict

def draw_annotated_image_placeholder(image_cv):
    """Returns the processed image for display."""
    # Since preprocessing is removed, we just return the image ready for display.
    return image_cv.copy()


# --- 6. UI and Execution Flow (Updated) ---

# Simplified function signature, removing all preprocessing parameters
def process_image_and_display(original_image_pil, unique_key_suffix):
    """
    Performs image processing, AI OCR, extraction, and displays results. 
    """
    st.subheader("Processing Image...")
    
    # Convert PIL image once for display
    visual_processed_img_cv = handle_pil_to_cv2(original_image_pil)

    with st.spinner("Running AI OCR and Structured Extraction..."):
        time.sleep(1) 
        
        # 1. Run AI Structured Extraction
        raw_extracted_data = ai_extract_data_structured(original_image_pil)
        
        if raw_extracted_data is None:
             st.stop() 

        # 2. Prepare data for display/download
        txt_file, csv_file, doc_file, extracted_data = create_downloadable_files(raw_extracted_data)
        
        # 3. Create placeholder annotated image (no boxes)
        annotated_img = draw_annotated_image_placeholder(visual_processed_img_cv)
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Processed Image (AI Extracted Data)")
        st.caption("Note: Bounding boxes are not available in this OCR implementation.")
        
        # Ensure CV2 BGR is converted to RGB for Streamlit display
        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
             
        st.image(rgb_img, width='stretch') 
        
    with col2:
        st.header("Extraction Results")
        
        # --- Results Form ---
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key): 
            st.text_input("License No", value=extracted_data["License No"])
            st.text_input("Name", value=extracted_data["Name"])
            st.text_input("NRC No", value=extracted_data["NRC No"])
            st.text_input("Date of Birth", value=extracted_data["Date of Birth"])
            st.text_input("Blood Type", value=extracted_data["Blood Type"])
            st.text_input("Valid Up To", value=extracted_data["Valid Up"])
            st.form_submit_button("Acknowledge") 
            
        st.subheader("Download Data")
        
        # --- Download Buttons ---
        st.download_button(
            label="⬇️ Download CSV", 
            data=csv_file, 
            file_name="license_data.csv", 
            mime="text/csv", 
            key=f"download_csv_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Download Plain Text", 
            data=txt_file, 
            file_name="license_data.txt", 
            mime="text/plain", 
            key=f"download_txt_{unique_key_suffix}" 
        )
        st.download_button(
            label="⬇️ Download Word (.doc)", 
            data=doc_file, 
            file_name="license_data.doc", 
            mime="application/msword", 
            key=f"download_doc_{unique_key_suffix}" 
        )

# --- Main App Body ---

st.title("🪪 Myanmar License Extractor (AI OCR)")
st.warning("Ensure your **GEMINI_API_KEY** is set in a **.env** file in the same directory and you have installed **python-dotenv** (`pip install python-dotenv`).")
st.write("This application uses an **AI model** for robust, structured data extraction.")

# --- Image Processing Settings (REMOVED) ---
# The st.expander for image enhancement settings is removed entirely.

# --- Tab Setup ---
tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

current_time_suffix = str(time.time()).replace('.', '') 

# --- Live Capture Tab ---
with tab1:
    st.header("Live Document Capture")
    captured_file = st.camera_input("Place the license clearly in the frame and click 'Take Photo'", key="camera_input")
    
    if captured_file is not None:
        image_pil = handle_file_to_pil(captured_file)
        
        if image_pil is not None:
            # Removed preprocessing parameters from the function call
            process_image_and_display(
                image_pil, 
                f"live_{current_time_suffix}"
            )
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

# --- Upload File Tab ---
with tab2:
    st.header("Upload Image File")
    uploaded_file = st.file_uploader("Upload License Image", type=['jpg', 'png', 'jpeg'], key="file_uploader")
    
    if uploaded_file is not None:
        image_pil = handle_file_to_pil(uploaded_file)
        
        if image_pil is not None:
            # Removed preprocessing parameters from the function call
            process_image_and_display(
                image_pil, 
                f"upload_{current_time_suffix}"
            )
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
