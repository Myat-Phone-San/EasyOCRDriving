import easyocr
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
    page_title="🪪 Myanmar Driving License Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Core OCR Engine ---
@st.cache_resource
def load_ocr_reader():
    """Loads and caches the EasyOCR reader (English only, since the labels are in English)."""
    # Note: Using 'en' only speeds up processing significantly compared to multi-language.
    # Myanmar (Burmese) script is not officially supported by EasyOCR.
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

# --- 2. Helper Functions ---

def get_box_props(bbox):
    """Calculates properties of a bounding box for spatial analysis."""
    x_min = min(p[0] for p in bbox)
    x_max = max(p[0] for p in bbox)
    y_min = min(p[1] for p in bbox)
    y_max = max(p[1] for p in bbox)
    return x_min, y_min, x_max, y_max, (x_min + x_max)/2, (y_min + y_max)/2

def parse_date(text):
    """Attempts to parse a date string from OCR text using common formats."""
    # Matches DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY
    match = re.search(r'\b(\d{1,2}[-\./]\d{1,2}[-\./]\d{4})\b', text)
    if match:
        d_str = match.group(1).replace(".", "-").replace("/", "-")
        try:
            # Assumes Day-Month-Year format typical of official documents
            return datetime.strptime(d_str, "%d-%m-%Y"), match.group(1)
        except ValueError:
            pass # Failed to parse
    return None, None

def is_valid_blood_type(text):
    """Validates and cleans extracted text to identify a blood type."""
    # Standard cleanup: replace 0 with O, remove punctuation
    clean = text.upper().replace("0", "O").replace(".", "").replace(",", "").strip() 
    if clean in ["A", "B", "AB", "O", "A+", "B+", "AB+", "O+"]:
        return clean
    return None

def merge_boxes(box_list):
    """Merges multiple bounding boxes into a single larger box."""
    if not box_list: return None
    all_x = [p[0] for box in box_list for p in box]
    all_y = [p[1] for box in box_list for p in box]
    
    x_min, y_min = min(all_x), min(all_y)
    x_max, y_max = max(all_x), max(all_y)

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def handle_file_to_cv2(uploaded_file):
    """Converts uploaded file or bytes to a CV2 image object."""
    if uploaded_file is None:
        return None
        
    # Read bytes from file-like object or directly from bytes
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_cv is None or image_cv.size == 0:
        return None
        
    return image_cv

def create_downloadable_files(extracted_data):
    """Generates content for CSV, TXT, and DOC download buttons."""
    results_dict = {k: v['text'] for k, v in extracted_data.items()}
    
    # TXT content
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    
    # CSV content
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # DOC content (simple tab-separated text, but with a .doc extension)
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    
    return txt_content, csv_content, doc_content

# --- 3. Main Extraction Logic ---
def extract_data_robust(raw_results):
    """
    Analyzes OCR results to robustly extract specific fields from a Myanmar
    Driving License, relying on keyword anchors and spatial relationships.
    """
    data = {
        "License No": {"text": "", "box": None, "anchor_box": None},
        "Name": {"text": "", "box": None, "anchor_box": None},
        "NRC No": {"text": "", "box": None, "anchor_box": None},
        "Date of Birth": {"text": "", "box": None, "anchor_box": None}, 
        "Blood Type": {"text": "", "box": None, "anchor_box": None},
        "Valid Up": {"text": "", "box": None, "anchor_box": None} 
    }

    clean_blocks = []
    all_dates = [] 

    # 1. Pre-process and categorize all text blocks
    for item in raw_results:
        bbox, text, prob = item
        if len(text.strip()) < 1: continue
        
        dt_obj, dt_str = parse_date(text)
        if dt_obj:
            all_dates.append({'obj': dt_obj, 'text': dt_str, 'box': bbox})

        clean_blocks.append({
            "box": bbox,
            "text": text,
            "clean": text.lower().replace(".", "").strip(),
            "props": get_box_props(bbox)
        })

    # 2. Extract Dates (Assuming DOB is chronologically first, Valid Up is last)
    if all_dates:
        all_dates.sort(key=lambda x: x['obj'])
        
        # Date of Birth (Earliest date)
        data["Date of Birth"]["text"] = all_dates[0]['text']
        data["Date of Birth"]["box"] = all_dates[0]['box']
        
        # Valid Up (Latest date, if more than one date exists)
        if len(all_dates) > 1:
            data["Valid Up"]["text"] = all_dates[-1]['text']
            data["Valid Up"]["box"] = all_dates[-1]['box']

    # 3. Find Anchor Blocks (Labels)
    anchors_map = {
        "License No": ["license", "no", "b/"],
        "Name": ["name"],
        "NRC No": ["nrc"],
        "Date of Birth": ["date of birth", "dob"],
        "Valid Up": ["valid", "up", "expiry"],
        "Blood Type": ["blood", "type", "blood type"] 
    }

    for field, keywords in anchors_map.items():
        anchor_block = next((block for block in clean_blocks if any(k in block["clean"] for k in keywords)), None)
        if anchor_block and data[field]["anchor_box"] is None:
             # Exclude "NRC No" if the block contains "nrc" but we are looking for "License No"
            if field == "License No" and "nrc" in anchor_block["clean"]: continue
            data[field]["anchor_box"] = anchor_block["box"]

            # 4. Use Anchor to find nearby Data (Spatial Search)
            ax_min, ay_min, ax_max, ay_max, ax_center, ay_center = anchor_block["props"]
            candidates = []
            
            for block in clean_blocks:
                # Skip the anchor itself and known date blocks
                if block == anchor_block or block["box"] == data["Date of Birth"]["box"] or block["box"] == data["Valid Up"]["box"]: continue
                
                bx_min, by_min, bx_max, by_max, bx_center, by_center = block["props"]

                # Check if the candidate text block is generally to the right of the label
                is_right = bx_min > ax_min - 10 
                
                # Check for vertical alignment (crucial for finding associated value)
                if field == "Blood Type":
                    # Blood type is often short and may not align perfectly vertically; use a larger margin
                    vert_aligned = (by_center > ay_min) and (by_center < ay_max + 40)
                else:
                    # Standard fields (Name, License, NRC) should be tightly aligned
                    vert_aligned = abs(ay_center - by_center) < 25

                if is_right and vert_aligned:
                    candidates.append(block)

            if candidates:
                candidates.sort(key=lambda b: b["props"][0]) # Sort by X-coordinate (left to right)
                
                if field == "Blood Type":
                    for c in candidates:
                        bt = is_valid_blood_type(c["text"])
                        if bt:
                            data[field]["text"] = bt
                            data[field]["box"] = c["box"]
                            break
                            
                elif field == "License No":
                    full_str = " ".join([c["text"] for c in candidates])
                    # Specific pattern: A/123456/12 or AA/12345/123
                    match = re.search(r'([A-Z]{1,2}/[\d]{4,6}/[\d]{2,4})', full_str)
                    if match:
                        data[field]["text"] = match.group(1)
                        # Find the boxes that constitute the license number
                        data[field]["box"] = merge_boxes([c["box"] for c in candidates if match.group(1) in c["text"] or any(part in c["text"] for part in match.group(1).split('/'))])
                    else:
                        data[field]["text"] = full_str
                        data[field]["box"] = merge_boxes([c["box"] for c in candidates])

                elif field in ["Name", "NRC No"]: 
                    # Filter out Burmese text which might confuse name/NRC
                    valid_c = [c for c in candidates if not any('\u1000' <= char <= '\u109f' for char in c["text"])]
                    if valid_c:
                        data[field]["text"] = " ".join([c["text"] for c in valid_c]).strip()
                        data[field]["box"] = merge_boxes([c["box"] for c in valid_c])

    # 5. Final NRC cleanup and fallback search
    if not data["NRC No"]["text"]:
        for block in clean_blocks:
            # Pattern: 12/ABCDEF(N)123456
            nrc_match = re.search(r'(\d{1,2}/[A-Z]{6}\(N\)\d{6})', block["text"].replace(" ", ""))
            if nrc_match:
                data["NRC No"]["text"] = nrc_match.group(1)
                data["NRC No"]["box"] = block["box"]
                break
                
    if data["NRC No"]["text"]:
        data["NRC No"]["text"] = data["NRC No"]["text"].upper().replace(" ", "").replace("-", "")

    return data

# --- 4. Visualization ---
def draw_annotated_image(image_cv, extracted_data):
    """Draws bounding boxes and labels on the image for visualization."""
    img_out = image_cv.copy()
    font_scale = 0.8
    font_thickness = 2
    
    # Define colors
    ANCHOR_COLOR = (255, 100, 0) # Orange/Blue for Anchor (Label)
    VALUE_COLOR = (0, 200, 0)   # Green for Value (Extracted Data)
    
    for field, info in extracted_data.items():
        # Draw Anchor Box (Label)
        if info['anchor_box'] is not None:
            tl, tr, br, bl = info['anchor_box']
            cv2.rectangle(img_out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), ANCHOR_COLOR, 2)
            
        # Draw Value Box (Extracted Data)
        if info['box'] is not None:
            tl, tr, br, bl = info['box']
            cv2.rectangle(img_out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), VALUE_COLOR, 3)
            # Add text label above the value box
            cv2.putText(img_out, field, (int(tl[0]), int(tl[1])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, VALUE_COLOR, font_thickness)
            
    return img_out

# --- 5. UI and Execution Flow ---

def process_image_and_display(image_cv, unique_key_suffix):
    """
    Performs OCR, extraction, and displays results in a two-column layout.
    Uses unique_key_suffix to prevent Streamlit ID collisions.
    """
    st.markdown("---")
    st.subheader("Extraction in Progress...")
    
    with st.spinner("Running EasyOCR and custom extraction logic..."):
        # Run OCR
        raw_results = reader.readtext(image_cv)
        # Run extraction logic
        extracted_data = extract_data_robust(raw_results)
        # Create visualization
        annotated_img = draw_annotated_image(image_cv, extracted_data)
        # Prepare download data
        txt_file, csv_file, doc_file = create_downloadable_files(extracted_data)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Annotated Image")
        st.caption("Blue/Orange boxes indicate the label (anchor), Green boxes indicate the extracted value.")
        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption="Processed Image", use_container_width=True)
        
    with col2:
        st.header("Extraction Results (Editable)")
        
        # --- Results Form (Using unique key) ---
        # Using a form allows users to manually correct OCR errors before saving.
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key): 
            st.text_input("License No", value=extracted_data["License No"]["text"], key=f"lic_{unique_key_suffix}")
            st.text_input("Name", value=extracted_data["Name"]["text"], key=f"name_{unique_key_suffix}")
            st.text_input("NRC No", value=extracted_data["NRC No"]["text"], key=f"nrc_{unique_key_suffix}")
            st.text_input("Date of Birth", value=extracted_data["Date of Birth"]["text"], key=f"dob_{unique_key_suffix}")
            st.text_input("Blood Type", value=extracted_data["Blood Type"]["text"], key=f"blood_{unique_key_suffix}")
            st.text_input("Valid Up To", value=extracted_data["Valid Up"]["text"], key=f"valid_{unique_key_suffix}")
            st.form_submit_button("Acknowledge & Confirm Data") 
            
        st.subheader("Download Extracted Data")
        
        # --- Download Buttons (Using unique key) ---
        
        # 1. CSV Button
        st.download_button(
            label="⬇️ Download as CSV", 
            data=csv_file, 
            file_name="license_data.csv", 
            mime="text/csv", 
            key=f"download_csv_{unique_key_suffix}"
        )
        
        # 2. Plain Text Button
        st.download_button(
            label="⬇️ Download as Plain Text", 
            data=txt_file, 
            file_name="license_data.txt", 
            mime="text/plain", 
            key=f"download_txt_{unique_key_suffix}"
        )
        
        # 3. Word/DOC Button
        st.download_button(
            label="⬇️ Download as .doc (Tabular)", 
            data=doc_file, 
            file_name="license_data.doc", 
            mime="application/msword", 
            key=f"download_doc_{unique_key_suffix}"
        )

# --- Main App Body ---

st.title("🪪 Myanmar License Extractor (OCR)")
st.caption("This tool uses EasyOCR and spatial logic to extract key fields from a Myanmar Driving License.")

tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

# Ensure a unique key is generated once per script run to prevent ID collision when switching tabs
current_time_suffix = str(time.time()).replace('.', '') 

with tab1:
    st.header("1. Live Document Capture")
    st.info("Place your Myanmar Driving License in the camera view and ensure good lighting.")
    
    col_cam, col_guide = st.columns([1, 1])

    with col_cam:
        captured_file = st.camera_input("Place the card in the center and click 'Take Photo'", key=f"camera_input_{current_time_suffix}")
        
    with col_guide:
        st.subheader("Framing Guide")
        st.markdown(
            """
            To get the best results:
            1. Hold the camera parallel to the card (avoid angles).
            2. Ensure all text on the card is sharp (in focus).
            3. Use strong, even lighting (no glare or shadows).
            """
        )
        #  is not precise enough.
        # Use a more relevant image query: 
        st.image("https://placehold.co/400x250/3c8d19/ffffff?text=Place+Card+in+Frame+Here", caption="Ideal Framing Example (Placeholder)")
    
    if captured_file is not None:
        image_cv = handle_file_to_cv2(captured_file)
        
        if image_cv is not None:
            process_image_and_display(image_cv, f"live_{current_time_suffix}")
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

with tab2:
    st.header("2. Upload Image File")
    
    uploaded_file = st.file_uploader(
        "Upload License Image (JPG or PNG)", 
        type=['jpg', 'png', 'jpeg'], 
        key=f"uploader_{current_time_suffix}"
    )
    
    if uploaded_file is not None:
        image_cv = handle_file_to_cv2(uploaded_file)
        
        if image_cv is not None:
            process_image_and_display(image_cv, f"upload_{current_time_suffix}")
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
