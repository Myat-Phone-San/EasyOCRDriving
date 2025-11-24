import os
# --- DOTENV IMPORT ---
from dotenv import load_dotenv 
# Load environment variables from a .env file (must be in the same directory)
load_dotenv()
# --- END DOTENV IMPORT ---

import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
import time 
import json # Used for loading JSON response from Gemini

# --- Configuration ---
st.set_page_config(
    page_title="🪪 Myanmar Driving License Extractor (AI OCR)",
    layout="wide"
)

from google import genai
from google.genai import types
from PIL import Image

# Initialize the Gemini Client
try:
    # The client automatically uses the GEMINI_API_KEY loaded by load_dotenv()
    client = genai.Client()
except Exception as e:
    st.error(f"Error initializing AI client. Please ensure your API key is set correctly in your .env file. Details: {e}")
    st.stop()


## 2. Data Extraction Prompt and Schema (Including Confidence Scores)

# Define the template for a single extracted field (value and confidence)
FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "confidence": {"type": "number", "format": "float", "description": "A confidence score between 0.0 (low) and 1.0 (high) for the extracted value."}
    },
    "required": ["value", "confidence"]
}

# Define the expected output structure (NESTED)
extraction_schema = {
    "type": "object",
    "properties": {
        "license_no": {**FIELD_SCHEMA, "description": "The driving license number, typically like 'A/123456/22', with its confidence."},
        "name": {**FIELD_SCHEMA, "description": "The full name of the license holder, with its confidence."},
        "nrc_no": {**FIELD_SCHEMA, "description": "The NRC ID number, typically like '12/MASANA(N)123456', with its confidence."},
        "date_of_birth": {**FIELD_SCHEMA, "description": "The date of birth in DD-MM-YYYY format, with its confidence."},
        "blood_type": {**FIELD_SCHEMA, "description": "The blood type, e.g., 'A+', 'B', 'O-', 'AB', with its confidence."},
        "valid_up": {**FIELD_SCHEMA, "description": "The license expiry date in DD-MM-YYYY format, with its confidence."}
    },
    "required": ["license_no", "name", "nrc_no", "date_of_birth", "blood_type", "valid_up"]
}

# The main prompt for the model
EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Driving License.
Extract the following key data fields: License No, Name, NRC No, Date of Birth, Blood Type, and Valid Up (Expiry Date).
For EACH field, you MUST provide the extracted 'value' and an objective 'confidence' score between 0.0 and 1.0 based on the clarity and certainty of the OCR result.
Return the result strictly as a JSON object matching the provided schema.
Ensure the extracted dates are in the DD-MM-YYYY format.
If a value is not found or is unreadable, return an empty string "" for the 'value' and a low confidence (e.g., 0.1).
Do not include any extra text or formatting outside of the JSON object.
"""

---

## 3. File Handling Function

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
        st.error(f"Error converting file to image: {e}")
        return None
        
# Removed handle_pil_to_cv2
# Removed preprocess_image

---

## 4. AI Extraction Logic

def run_structured_extraction(image_pil):
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
        structured_data = json.loads(response.text)
        return structured_data
        
    except genai.errors.APIError as e:
        st.error(f"AI API Error: Could not process the image. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

---

## 5. Helper Functions (Updated for Confidence)

def create_downloadable_files(extracted_dict):
    """Formats the extracted data (including confidence) into CSV, TXT, and DOC formats."""
    # Mapping from schema key to display name
    field_map = {
        'license_no': "License No",
        'name': "Name",
        'nrc_no': "NRC No",
        'date_of_birth': "Date of Birth",
        'blood_type': "Blood Type",
        'valid_up': "Valid Up"
    }
    
    results_list = [] # For DataFrame and TXT/DOC
    
    for key, display_name in field_map.items():
        data = extracted_dict.get(key, {})
        value = data.get('value', '')
        # Handle cases where confidence might be missing or not a float (default to 0.0)
        try:
            confidence = float(data.get('confidence', 0.0))
        except (ValueError, TypeError):
            confidence = 0.0
        
        results_list.append({
            "Field": display_name,
            "Value": value,
            "Confidence (%)": f"{confidence*100:.2f}%"
        })
        
    # TXT Content
    txt_content = "\n".join([
        f"{item['Field']}: {item['Value']} (Confidence: {item['Confidence (%)']})" 
        for item in results_list
    ])
    
    # CSV/DataFrame Content
    df = pd.DataFrame(results_list)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # DOC Content (simple tab-separated for copy-paste into Word)
    doc_content = "\n".join([
        f"{item['Field']}\t{item['Value']}\t{item['Confidence (%)']}" 
        for item in results_list
    ])
    
    return txt_content, csv_content, doc_content, extracted_dict


---

## 6. UI and Execution Flow (Updated for Confidence Display)

def process_image_and_display(original_image_pil, unique_key_suffix):
    """
    Performs AI extraction and displays results, including confidence scores. 
    """
    st.subheader("Processing Image...")
    
    # Use a placeholder for the spinner to make the UX cleaner
    with st.spinner("Running AI Structured Extraction..."):
        time.sleep(1) # Visual pause
        
        # 1. Run Structured Extraction
        raw_extracted_data = run_structured_extraction(original_image_pil)
        
        if raw_extracted_data is None:
             st.stop() 

        # 2. Prepare data for display/download
        txt_file, csv_file, doc_file, extracted_data = create_downloadable_files(raw_extracted_data)
        
    st.success("✅ Extraction Complete!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Uploaded Image")
        # Display the original PIL image directly
        st.image(original_image_pil, use_column_width=True) 
        
    with col2:
        st.header("Extraction Results")
        
        # --- Results Display with Confidence Scores ---
        st.markdown("---")
        
        # Define fields to display based on the raw extracted data keys
        fields_to_display = [
            ("License No", 'license_no'), 
            ("Name", 'name'), 
            ("NRC No", 'nrc_no'), 
            ("Date of Birth", 'date_of_birth'), 
            ("Blood Type", 'blood_type'), 
            ("Valid Up To", 'valid_up')
        ]
        
        # Use a DataFrame for a compact, readable table display of results
        results_df = pd.DataFrame([
            {
                "Field": display_name,
                "Value": extracted_data.get(key, {}).get('value', ''),
                "Confidence": f"{extracted_data.get(key, {}).get('confidence', 0.0)*100:.2f}%"
            }
            for display_name, key in fields_to_display
        ])
        
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        
        st.subheader("Download Data")
        
        # --- Download Buttons ---
        st.download_button(
            label="⬇️ Download CSV (with Confidence)", 
            data=csv_file, 
            file_name="license_data_with_confidence.csv", 
            mime="text/csv", 
            key=f"download_csv_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Download Plain Text (with Confidence)", 
            data=txt_file, 
            file_name="license_data_with_confidence.txt", 
            mime="text/plain", 
            key=f"download_txt_{unique_key_suffix}" 
        )
        st.download_button(
            label="⬇️ Download Word (.doc) (Tab Separated)", 
            data=doc_file, 
            file_name="license_data_with_confidence.doc", 
            mime="application/msword", 
            key=f"download_doc_{unique_key_suffix}" 
        )

---

## 7. Main App Body

st.title("🪪 Myanmar License Extractor (AI OCR)")
st.caption("Powered by Google Gemini for Structured Data Extraction with Confidence Scoring.")

# --- Tab Setup ---
tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

current_time_suffix = str(time.time()).replace('.', '') 

# --- Live Capture Tab ---
with tab1:
    st.header("Live Document Capture")
    st.info("Ensure the license is well-lit and clearly focused for the best OCR results.")
    captured_file = st.camera_input("Place the license clearly in the frame and click 'Take Photo'", key="camera_input")
    
    if captured_file is not None:
        image_pil = handle_file_to_pil(captured_file)
        
        if image_pil is not None:
            process_image_and_display(
                image_pil, 
                f"live_{current_time_suffix}"
            )
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

# --- Upload File Tab ---
with tab2:
    st.header("Upload Image File")
    uploaded_file = st.file_uploader("Upload License Image (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'], key="file_uploader")
    
    if uploaded_file is not None:
        image_pil = handle_file_to_pil(uploaded_file)
        
        if image_pil is not None:
            process_image_and_display(
                image_pil, 
                f"upload_{current_time_suffix}"
            )
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
