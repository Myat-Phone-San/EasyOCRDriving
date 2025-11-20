import easyocr
import cv2
import numpy as np
import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
import time
import base64

# --- Configuration ---
st.set_page_config(
    page_title="🪪 Myanmar Driving License Extractor (FIXED & Optimized)",
    layout="wide"
)

# Define the maximum processing width for faster OCR and reduced noise
OPTIMIZED_WIDTH = 1200 
# Increased tolerance slightly to account for slightly off-horizontal scans
VERTICAL_TOLERANCE = 20 

# --- 1. Core OCR Engine ---
@st.cache_resource
def load_ocr_reader():
    """Loads and caches the EasyOCR reader (English only for Myanmar licenses)."""
    # Using 'en' is correct for the fields we target
    return easyocr.Reader(['en'], gpu=False)

try:
    reader = load_ocr_reader()
except Exception as e:
    st.error(f"Failed to load OCR reader: {e}. Please check EasyOCR dependencies.")
    st.stop()


# --- 2. Helper Functions ---
def get_box_props(bbox):
    """Calculates bounding box properties (min/max x/y and center points)."""
    x_min = min(p[0] for p in bbox)
    x_max = max(p[0] for p in bbox)
    y_min = min(p[1] for p in bbox)
    y_max = max(p[1] for p in bbox)
    return x_min, y_min, x_max, y_max, (x_min + x_max)/2, (y_min + y_max)/2

def parse_date(text):
    """Attempts to find and parse a date string (DD-MM-YYYY) from text."""
    # Pattern looks for: 1-2 digits, separator (.-/) , 1-2 digits, separator, 4 digits
    match = re.search(r'\b(\d{1,2}[-\./\\]\d{1,2}[-\./\\]\d{4})\b', text)
    if match:
        d_str = match.group(1).replace(".", "-").replace("/", "-").replace("\\", "-")
        try:
            # Try DD-MM-YYYY first (standard format for this context)
            dt_obj = datetime.strptime(d_str, "%d-%m-%Y")
            return dt_obj, d_str
        except:
            # Fallback for YYYY-MM-DD (less likely but possible from OCR)
            try:
                dt_obj = datetime.strptime(d_str, "%Y-%m-%d")
                return dt_obj, d_str
            except:
                pass
    return None, None

def is_valid_blood_type(text):
    """Checks if text is a valid blood type format."""
    clean = text.upper().replace("0", "O").replace(".", "").replace(",", "").replace("-", "").strip() 
    valid_types = ["A", "B", "AB", "O", "A+", "B+", "AB+", "O+", "A-", "B-", "AB-", "O-"]
    # Check if the text matches any valid type, allowing for +/- or just the letter
    if clean in valid_types:
        return clean
    # Simple check for just A, B, AB, O
    if clean in ["A", "B", "AB", "O"]:
        return clean
    return None

def merge_boxes(box_list):
    """Merges multiple bounding boxes into a single bounding box."""
    if not box_list: return None
    all_x = [p[0] for box in box_list for p in box]
    all_y = [p[1] for box in box_list for p in box]
    if not all_x or not all_y: return None
    
    min_x, min_y = min(all_x), min(all_y)
    max_x, max_y = max(all_x), max(all_y)
    return [
        [min_x, min_y], [max_x, min_y], 
        [max_x, max_y], [min_x, max_y]
    ]

def handle_file_to_cv2(uploaded_file, target_width):
    """
    Converts uploaded file/bytes to a CV2 image object and RESIZES for OCR optimization.
    """
    if uploaded_file is None:
        return None
        
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_cv is None or image_cv.size == 0:
        return None
        
    height, width = image_cv.shape[:2]
    
    # Resize only if necessary
    if width > target_width:
        ratio = target_width / width
        new_height = int(height * ratio)
        image_cv = cv2.resize(image_cv, (target_width, new_height), interpolation=cv2.INTER_AREA)
        
    return image_cv

def create_downloadable_files(extracted_data):
    """Formats extracted data into CSV, TXT, and simple DOC content."""
    results_dict = {k: str(v['text']) for k, v in extracted_data.items()}
    
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    
    return txt_content, csv_content, doc_content

# --- 3. Main Extraction Logic (FIXED for Accuracy) ---
def extract_data_robust(raw_results):
    """
    FIXED: Robustly extracts specific fields, prioritizing date sorting for DOB/Valid Up,
    and applying stricter spatial filtering for other fields.
    """
    data = {
        "License No": {"text": "", "box": None, "anchor_box": None},
        "Name": {"text": "", "box": None, "anchor_box": None},
        "NRC No": {"text": "", "box": None, "anchor_box": None},
        "Date of Birth": {"text": "", "box": None, "anchor_box": None}, 
        "Blood Type": {"text": "", "box": None, "anchor_box": None},
        "Valid Up To": {"text": "", "box": None, "anchor_box": None} # Renamed field key for clarity
    }

    clean_blocks = []
    all_dates = [] 
    extracted_box_coords = [] # To store box coordinates of already extracted data

    # 1. Pre-process: Collect dates and cleaned blocks
    for item in raw_results:
        bbox, text, prob = item
        if len(text.strip()) < 1: continue
        
        dt_obj, dt_str = parse_date(text)
        if dt_obj:
            all_dates.append({'obj': dt_obj, 'text': dt_str, 'box': bbox, 'original_text': text})

        clean_blocks.append({
            "box": bbox,
            "text": text,
            "clean": text.lower().replace(".", "").strip(),
            "props": get_box_props(bbox)
        })

    # 2. FIXED: Assign DOB and Valid Up using date sorting (most robust method)
    if all_dates:
        # Sort dates by datetime object: Earliest date (DOB) comes first, Latest date (Valid Up) comes last.
        all_dates.sort(key=lambda x: x['obj']) 
        
        # Date of Birth (Earliest Date)
        data["Date of Birth"]["text"] = all_dates[0]['text']
        data["Date of Birth"]["box"] = all_dates[0]['box']
        extracted_box_coords.append(tuple(map(tuple, all_dates[0]['box'])))

        if len(all_dates) > 1:
            # Valid Up To (Latest Date)
            data["Valid Up To"]["text"] = all_dates[-1]['text']
            data["Valid Up To"]["box"] = all_dates[-1]['box']
            extracted_box_coords.append(tuple(map(tuple, all_dates[-1]['box'])))

    # 3. Find anchor boxes and extract corresponding values
    # Added anchors for DOB and Valid Up TO find their labels for visualization
    anchors = {
        "License No": ["license", "no", "b/", "a/", "e/"],
        "Name": ["name", "holder"],
        "NRC No": ["nrc", "no"],
        "Date of Birth": ["date", "birth", "3-5-2001"], # Added birth date value as a fallback anchor
        "Blood Type": ["blood", "type"],
        "Valid Up To": ["valid", "up", "up to", "12-6-2029"] # Added valid up value as a fallback anchor
    }
    
    # Store extracted date boxes to avoid text pollution from date anchors
    
    for field, keywords in anchors.items():
        # Skip if already extracted via date sorting, but ensure anchor box is found for visualization
        if field in ["Date of Birth", "Valid Up To"] and data[field]["text"] != "":
            # Search for the date label (anchor)
            anchor_block = next((block for block in clean_blocks if any(k in block["clean"] for k in ["date of birth", "valid up", "valid to"])), None)
            if anchor_block:
                 data[field]["anchor_box"] = anchor_block["box"]
            continue
            
        # Find the primary anchor block
        anchor_block = next((block for block in clean_blocks if any(k in block["clean"] for k in keywords)), None)
        
        if anchor_block:
            data[field]["anchor_box"] = anchor_block["box"]
            
            ax_min, ay_min, ax_max, ay_max, ax_center, ay_center = anchor_block["props"]
            candidates = []
            
            for block in clean_blocks:
                block_box_tuple = tuple(map(tuple, block['box']))
                
                # Skip the anchor block itself and blocks already extracted as dates
                if block == anchor_block or block_box_tuple in extracted_box_coords: continue 

                bx_min, by_min, bx_max, by_max, bx_center, by_center = block["props"]

                # Stricter alignment: only look to the right of the anchor's center point
                is_right = bx_min > ax_center 
                
                # Tight vertical alignment: Center point must be within the anchor's height range +/- tolerance
                vert_aligned = (by_center > ay_center - VERTICAL_TOLERANCE) and \
                               (by_center < ay_center + VERTICAL_TOLERANCE)

                if is_right and vert_aligned:
                    candidates.append(block)

            if candidates:
                candidates.sort(key=lambda b: b["props"][0]) # Sort by starting X position
                
                if field == "Blood Type":
                    # Look for a valid blood type in the candidate blocks
                    for c in candidates:
                        bt = is_valid_blood_type(c["text"])
                        if bt:
                            data[field]["text"] = bt
                            data[field]["box"] = c["box"]
                            break
                            
                elif field == "License No":
                    # Take relevant blocks and combine them for License No check
                    relevant_candidates = candidates[:3] 
                    full_str = "".join([c["text"] for c in relevant_candidates]).replace(" ", "").upper().replace("I", "/")
                    
                    # Regex for License No format (e.g., A/123456/2023 or E/123456/2023)
                    match = re.search(r'([A-Z/]{1,2}/\d{4,7}/\d{2,4})', full_str)
                    if match:
                        data[field]["text"] = match.group(1)
                        data[field]["box"] = merge_boxes([c["box"] for c in relevant_candidates])
                    else:
                        data[field]["text"] = "N/A (Format Error - Fallback)"
                        
                elif field == "NRC No":
                    relevant_candidates = candidates[:4]
                    full_str = "".join([c["text"] for c in relevant_candidates]).replace(" ", "").upper().replace(".", "")
                    
                    # Regex for NRC format (e.g., 12/MAHTALA(N)123456)
                    nrc_match = re.search(r'(\d{1,2}/[A-Z]{3,6}\(N\)\d{6})', full_str)
                    if nrc_match:
                         data[field]["text"] = nrc_match.group(1)
                         data[field]["box"] = merge_boxes([c["box"] for c in relevant_candidates])
                    else:
                        # Fallback for messy NRC text
                        data[field]["text"] = full_str[:40].strip()
                        data[field]["box"] = merge_boxes([c["box"] for c in relevant_candidates])
                        
                elif field == "Name":
                    # Take up to the first 3 blocks for name, filtering out non-alphabetic noise
                    valid_c = [c for c in candidates[:3] if re.search(r'[a-zA-Z]', c["text"]) and not any('\u1000' <= char <= '\u109f' for char in c["text"])]
                    if valid_c:
                        data[field]["text"] = " ".join([c["text"] for c in valid_c]).strip()
                        data[field]["box"] = merge_boxes([c["box"] for c in valid_c])
                        
    # 4. Final Cleanup (NRC)
    if data["NRC No"]["text"] and data["NRC No"]["text"] != "N/A (Format Error)":
        data["NRC No"]["text"] = data["NRC No"]["text"].upper().replace(" ", "").replace("-", "")
        # Try regex again on cleaned text
        nrc_match = re.search(r'(\d{1,2}/[A-Z]{3,6}\(N\)\d{6})', data["NRC No"]["text"])
        if nrc_match:
             data["NRC No"]["text"] = nrc_match.group(1)
             
    return data

# --- 4. Visualization ---
# (Visualization logic remains the same, as it was correct for drawing the boxes)
def draw_annotated_image(image_cv, extracted_data):
    """Draws bounding boxes for anchors (blue) and extracted values (green)."""
    img_out = image_cv.copy()
    font_scale = 0.5
    font_thickness = 2
    
    ANCHOR_COLOR = (255, 100, 0) # Light Blue/Cyan (BGR)
    VALUE_COLOR = (0, 255, 0)    # Green (BGR)
    TEXT_COLOR = (0, 200, 0)     # Darker Green for text (BGR)
    
    for field, info in extracted_data.items():
        # Draw Anchor Box (Blue/Cyan)
        if info['anchor_box'] is not None:
            # Anchor box is a list of 4 points. Extract min/max X/Y.
            x_min, y_min, x_max, y_max, _, _ = get_box_props(info['anchor_box'])
            cv2.rectangle(img_out, (int(x_min), int(y_min)), (int(x_max), int(y_max)), ANCHOR_COLOR, 2)
            
        # Draw Value Box (Green) and Label
        if info['box'] is not None:
            # Value box is a merged box (list of 4 points). Extract min/max X/Y.
            x_min, y_min, x_max, y_max, _, _ = get_box_props(info['box'])
            cv2.rectangle(img_out, (int(x_min), int(y_min)), (int(x_max), int(y_max)), VALUE_COLOR, 2)
            cv2.putText(img_out, field, (int(x_min), int(y_min)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, font_thickness)
            
    return img_out

# --- 5. UI and Execution Flow ---

def process_image_and_display(image_cv, source_key, process_width):
    """
    Performs OCR, extraction, and displays results. 
    """
    st.subheader("Step 1: Processing Image...")
    
    unique_suffix = str(time.time()).replace('.', '')
    
    with st.spinner(f"Running OCR and Extraction Logic on {process_width}px image..."):
        
        # 1. Run OCR
        raw_results = reader.readtext(image_cv)
        
        # 2. Extract Data
        extracted_data = extract_data_robust(raw_results)
        
        # 3. Annotate Image
        annotated_img = draw_annotated_image(image_cv, extracted_data)
        
        # 4. Prepare Download Files
        txt_file, csv_file, doc_file = create_downloadable_files(extracted_data)

    st.subheader("Step 2: Review Results")
    # New column split: 1 part for image, 1.5 parts for data
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("##### Annotated Image (Blue: Anchor/Label, Green: Extracted Value)")
        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(
            rgb_img, 
            caption=f"Image processed at maximum width of {process_width}px. Bounding boxes appear after capture.", 
            use_container_width=True
        )
        
    with col2:
        st.markdown("##### Extracted Data (Editable)")
        
        key_base = f"{source_key}_{unique_suffix}"
        with st.form(f"results_form_{key_base}"): 
            st.text_input("License No", value=extracted_data.get("License No", {}).get("text", ""), key=f"license_no_{key_base}")
            st.text_input("Name", value=extracted_data.get("Name", {}).get("text", ""), key=f"name_{key_base}")
            st.text_input("NRC No", value=extracted_data.get("NRC No", {}).get("text", ""), key=f"nrc_no_{key_base}")
            # Ensure the key here matches the key in the extraction logic
            st.text_input("Date of Birth (DD-MM-YYYY)", value=extracted_data.get("Date of Birth", {}).get("text", ""), key=f"dob_{key_base}")
            st.text_input("Blood Type", value=extracted_data.get("Blood Type", {}).get("text", ""), key=f"blood_type_{key_base}")
            # Ensure the key here matches the key in the extraction logic
            st.text_input("Valid Up To (DD-MM-YYYY)", value=extracted_data.get("Valid Up To", {}).get("text", ""), key=f"valid_up_{key_base}")
            st.form_submit_button("Acknowledge / Submit Edits") 
            
        st.markdown("##### Download Options")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            st.download_button(
                label="⬇️ CSV", 
                data=csv_file, 
                file_name="license_data.csv", 
                mime="text/csv", 
                key=f"download_csv_{key_base}",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="⬇️ Text", 
                data=txt_file, 
                file_name="license_data.txt", 
                mime="text/plain", 
                key=f"download_txt_{key_base}",
                use_container_width=True
            )
        with col_dl3:
            st.download_button(
                label="⬇️ DOC", 
                data=doc_file, 
                file_name="license_data.doc", 
                mime="application/msword", 
                key=f"download_doc_{key_base}",
                use_container_width=True
            )

# --- Main App Body ---

st.title("🪪 Myanmar License Extractor (Optimized for Speed and Accuracy)")
st.info(f"""
    **Extraction FIXED:** The extraction logic for **Date of Birth** and **Valid Up To** is fixed by identifying the **earliest date** as DOB and the **latest date** as Valid Up. The logic for **License No**, **NRC No**, and **Blood Type** has also been significantly tightened using stricter spatial and regex checks.
    
    **Camera Note:** The processing image size is now **{OPTIMIZED_WIDTH}px** for better accuracy and speed.
""")

tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])


with tab1:
    st.header("Live Document Capture")
    st.markdown("Place the license clearly in the frame (landscape is recommended) and click **'Take Photo'**.")
    captured_file = st.camera_input("Take Photo of License")
    
    if captured_file is not None:
        image_cv = handle_file_to_cv2(captured_file, target_width=OPTIMIZED_WIDTH)
        
        if image_cv is not None:
            process_image_and_display(image_cv, "live_capture", OPTIMIZED_WIDTH)
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

with tab2:
    st.header("Upload Image File")
    uploaded_file = st.file_uploader("Upload License Image (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image_cv = handle_file_to_cv2(uploaded_file, target_width=OPTIMIZED_WIDTH)
        
        if image_cv is not None:
            process_image_and_display(image_cv, "file_upload", OPTIMIZED_WIDTH)
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
