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

# --- Configuration ---
st.set_page_config(
    page_title="🪪 Myanmar Driving License Extractor (AI OCR)",
    layout="wide"
)

from google import genai
from google.genai import types
from PIL import Image
import json # Moved json import up for use in the extraction function

# Initialize the Gemini Client
try:
    # The client automatically uses the GEMINI_API_KEY loaded by load_dotenv()
    client = genai.Client()
except Exception as e:
    # Changed output text
    st.error(f"Error initializing AI client. Please ensure your API key is set correctly in your .env file. Details: {e}")
    st.stop()


# --- 2. Data Extraction Prompt and Schema (MODIFIED) ---

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

# The main prompt for the model (MODIFIED)
EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Driving License.
Extract the following key data fields: License No, Name, NRC No, Date of Birth, Blood Type, and Valid Up (Expiry Date).
For EACH field, you MUST provide the extracted 'value' and an objective 'confidence' score between 0.0 and 1.0 based on the clarity and certainty of the OCR result.
Return the result strictly as a JSON object matching the provided schema.
Ensure the extracted dates are in the DD-MM-YYYY format.
If a value is not found or is unreadable, return an empty string "" for the 'value' and a low confidence (e.g., 0.1).
Do not include any extra text or formatting outside of the JSON object.
"""

# --- 3. File Handling Function (Only PIL remains) ---
# ... (handle_file_to_pil remains the same) ...
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
# ... (end of handle_file_to_pil) ...

# --- 4. AI Extraction Logic ---
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
        # Changed output text
        st.error(f"AI API Error: Could not process the image. Details: {e}")
        return None
    except Exception as e:
        # Changed output text
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

# --- 5. Helper Functions (MODIFIED) ---

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
    
    # Prepare the dictionary for display/download with confidence
    results_dict_flat = {}
    results_list = [] # For DataFrame
    
    for key, display_name in field_map.items():
        data = extracted_dict.get(key, {})
        value = data.get('value', '')
        confidence = data.get('confidence', 0.0)
        
        # Store for Streamlit display
        results_dict_flat[display_name] = value
        
        # Store for DataFrame/Download
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


# --- 6. UI and Execution Flow (Updated for Confidence Display) ---

# Simplified function signature - removed grayscale, contrast, brightness, denoise_h
def process_image_and_display(original_image_pil, unique_key_suffix):
    """
    Performs AI extraction and displays results. 
    """
    st.subheader("Processing Image...")
    
    with st.spinner("Running AI Structured Extraction..."):
        time.sleep(1) 
        
        # 1. Run Structured Extraction
        raw_extracted_data = run_structured_extraction(original_image_pil)
        
        if raw_extracted_data is None:
             st.stop() 

        # 2. Prepare data for display/download
        txt_file, csv_file, doc_file, extracted_data = create_downloadable_files(raw_extracted_data)
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Uploaded Image")
        # Display the original PIL image directly
        st.image(original_image_pil, width='stretch') 
        
    with col2:
        st.header("Extraction Results")
        
        # --- Results Display (Using Columns for better layout of value and confidence) ---
        st.markdown("**Value** | **Confidence (%)**")
        st.markdown("---")

        fields_to_display = [
            ("License No", 'license_no'), 
            ("Name", 'name'), 
            ("NRC No", 'nrc_no'), 
            ("Date of Birth", 'date_of_birth'), 
            ("Blood Type", 'blood_type'), 
            ("Valid Up To", 'valid_up')
        ]
        
        # Using a standard display rather than a form to show confidence
        for display_name, key in fields_to_display:
            value = extracted_data.get(key, {}).get('value', '')
            confidence = extracted_data.get(key, {}).get('confidence', 0.0)
            
            # Format confidence as a percentage string
            confidence_str = f"{confidence*100:.2f}%"
            
            st.markdown(f"**{display_name}:**")
            col_val, col_conf = st.columns([2, 1])
            col_val.code(value, language='text')
            col_conf.code(confidence_str, language='text')
            
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

# --- Main App Body ---
# ... (The rest of the main app body remains the same, calling the updated process_image_and_display) ...

st.title("🪪 Myanmar License Extractor (AI OCR)")

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
            process_image_and_display(
                image_pil, 
                f"upload_{current_time_suffix}"
            )
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
