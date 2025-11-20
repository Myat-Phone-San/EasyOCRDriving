import easyocr
import cv2
import numpy as np
import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
import time # Used for generating unique form/element keys

# --- Configuration ---
st.set_page_config(
    page_title="🪪 Myanmar Driving License Extractor (Live & Upload)",
    layout="wide"
)

# --- 1. Core OCR Engine ---
@st.cache_resource
def load_ocr_reader():
    """Loads and caches the EasyOCR reader."""
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

# --- 2. Helper Functions (Unchanged) ---
def get_box_props(bbox):
    x_min = min(p[0] for p in bbox)
    x_max = max(p[0] for p in bbox)
    y_min = min(p[1] for p in bbox)
    y_max = max(p[1] for p in bbox)
    return x_min, y_min, x_max, y_max, (x_min + x_max)/2, (y_min + y_max)/2

def parse_date(text):
    match = re.search(r'\b(\d{1,2}[-\./]\d{1,2}[-\./]\d{4})\b', text)
    if match:
        d_str = match.group(1).replace(".", "-").replace("/", "-")
        try:
            return datetime.strptime(d_str, "%d-%m-%Y"), match.group(1)
        except:
            pass
    return None, None

def is_valid_blood_type(text):
    clean = text.upper().replace("0", "O").replace(".", "").replace(",", "").strip() 
    if clean in ["A", "B", "AB", "O", "A+", "B+", "AB+", "O+"]:
        return clean
    return None

def merge_boxes(box_list):
    if not box_list: return None
    all_x = [p[0] for box in box_list for p in box]
    all_y = [p[1] for box in box_list for p in box]
    return [[min(all_x), min(all_y)], [max(all_x), min(all_y)], 
            [max(all_x), max(all_y)], [min(all_x), max(all_y)]]

def handle_file_to_cv2(uploaded_file):
    """Converts uploaded file or bytes to a CV2 image object."""
    # Check if the uploaded_file is a file-like object or raw bytes
    if uploaded_file is None:
        return None
        
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check for empty/invalid image (e.g., if cv2 failed to decode)
    if image_cv is None or image_cv.size == 0:
        return None
        
    return image_cv

def create_downloadable_files(extracted_data):
    results_dict = {k: v['text'] for k, v in extracted_data.items()}
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    return txt_content, csv_content, doc_content

# --- 3. Main Logic (Skipped for brevity, as it's not the source of the error) ---
def extract_data_robust(raw_results):
    # ... (Your existing robust extraction logic remains here) ...
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

    if all_dates:
        all_dates.sort(key=lambda x: x['obj'])
        data["Date of Birth"]["text"] = all_dates[0]['text']
        data["Date of Birth"]["box"] = all_dates[0]['box']
        
        if len(all_dates) > 1:
            data["Valid Up"]["text"] = all_dates[-1]['text']
            data["Valid Up"]["box"] = all_dates[-1]['box']
        elif len(all_dates) == 1:
            data["Date of Birth"]["text"] = all_dates[0]['text']

    date_and_blood_anchors = {
        "Date of Birth": ["date of birth", "dob"],
        "Valid Up": ["valid", "up", "expiry"],
        "Blood Type": ["blood", "type", "blood type"] 
    }
    
    for field, keywords in date_and_blood_anchors.items():
        anchor_block = next((block for block in clean_blocks if any(k in block["clean"] for k in keywords)), None)
        if anchor_block:
            data[field]["anchor_box"] = anchor_block["box"]

    anchors = {
        "License No": ["license", "no", "b/"],
        "Name": ["name"],
        "NRC No": ["nrc"],
        "Blood Type": ["blood", "type"]
    }

    for field, keywords in anchors.items():
        if data[field]["text"] and field not in ["Blood Type", "License No"]:
            continue
            
        anchor_block = None
        for block in clean_blocks:
            if any(k in block["clean"] for k in keywords):
                if field == "License No" and "nrc" in block["clean"]: continue
                anchor_block = block
                if data[field]["anchor_box"] is None:
                    data[field]["anchor_box"] = anchor_block["box"]
                break
        
        if anchor_block:
            ax_min, ay_min, ax_max, ay_max, ax_center, ay_center = anchor_block["props"]
            candidates = []
            
            for block in clean_blocks:
                if block == anchor_block: continue
                if any(d['text'] in block["text"] for d in all_dates): continue
                
                bx_min, by_min, bx_max, by_max, bx_center, by_center = block["props"]

                is_right = bx_min > ax_min - 10 
                if field == "Blood Type":
                    vert_aligned = (by_center > ay_min) and (by_center < ay_max + 40)
                else:
                    vert_aligned = abs(ay_center - by_center) < 25

                if is_right and vert_aligned:
                    candidates.append(block)

            if candidates:
                candidates.sort(key=lambda b: b["props"][0])
                
                if field == "Blood Type":
                    for c in candidates:
                        bt = is_valid_blood_type(c["text"])
                        if bt:
                            data[field]["text"] = bt
                            data[field]["box"] = c["box"]
                            break
                            
                elif field == "License No":
                    full_str = " ".join([c["text"] for c in candidates])
                    match = re.search(r'([A-Z]{1,2}/[\d]{4,6}/[\d]{2,4})', full_str)
                    if match:
                        data[field]["text"] = match.group(1)
                        data[field]["box"] = merge_boxes([c["box"] for c in candidates if match.group(1) in c["text"]])
                    else:
                         data[field]["text"] = full_str
                         data[field]["box"] = merge_boxes([c["box"] for c in candidates])

                else: # Name, NRC
                    valid_c = [c for c in candidates if not any('\u1000' <= char <= '\u109f' for char in c["text"])]
                    if valid_c:
                        data[field]["text"] = " ".join([c["text"] for c in valid_c]).strip()
                        data[field]["box"] = merge_boxes([c["box"] for c in valid_c])

    if not data["NRC No"]["text"]:
        for block in clean_blocks:
            nrc_match = re.search(r'(\d{1,2}/[A-Z]{6}\(N\)\d{6})', block["text"].replace(" ", ""))
            if nrc_match:
                data["NRC No"]["text"] = nrc_match.group(1)
                data["NRC No"]["box"] = block["box"]
                break
                
    if data["NRC No"]["text"]:
        data["NRC No"]["text"] = data["NRC No"]["text"].upper().replace(" ", "").replace("-", "")

    return data
# --- 4. Visualization (Skipped for brevity) ---
def draw_annotated_image(image_cv, extracted_data):
    img_out = image_cv.copy()
    font_scale = 0.7
    font_thickness = 2
    
    for field, info in extracted_data.items():
        if info['anchor_box'] is not None:
            tl, tr, br, bl = info['anchor_box']
            cv2.rectangle(img_out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (255, 0, 0), 2)
            
        if info['box'] is not None:
            tl, tr, br, bl = info['box']
            cv2.rectangle(img_out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
            cv2.putText(img_out, field, (int(tl[0]), int(tl[1])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 0), font_thickness)
            
    return img_out

# --- 5. UI and Execution Flow (FIXED) ---

def process_image_and_display(image_cv, unique_key_suffix):
    """
    Performs OCR, extraction, and displays results. 
    Accepts unique_key_suffix to prevent ID collision.
    """
    st.subheader("Processing Image...")
    
    with st.spinner("Running OCR and Extraction Logic..."):
        time.sleep(1) 
        
        raw_results = reader.readtext(image_cv)
        extracted_data = extract_data_robust(raw_results)
        annotated_img = draw_annotated_image(image_cv, extracted_data)
        txt_file, csv_file, doc_file = create_downloadable_files(extracted_data)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Annotated Image (Blue: Label, Green: Value)")
        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, use_container_width=True)
        
    with col2:
        st.header("Extraction Results")
        
        # --- Results Form (Using unique key) ---
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key): 
            st.text_input("License No", value=extracted_data["License No"]["text"])
            st.text_input("Name", value=extracted_data["Name"]["text"])
            st.text_input("NRC No", value=extracted_data["NRC No"]["text"])
            st.text_input("Date of Birth", value=extracted_data["Date of Birth"]["text"])
            st.text_input("Blood Type", value=extracted_data["Blood Type"]["text"])
            st.text_input("Valid Up To", value=extracted_data["Valid Up"]["text"])
            st.form_submit_button("Acknowledge") 
            
        st.subheader("Download Data")
        
        # --- Download Buttons (Using unique key) ---
        
        # 1. CSV Button
        st.download_button(
            label="⬇️ Download CSV", 
            data=csv_file, 
            file_name="license_data.csv", 
            mime="text/csv", 
            help="Download data as a Comma Separated Values file.",
            key=f"download_csv_{unique_key_suffix}" # FIX APPLIED HERE
        )
        
        # 2. Plain Text Button
        st.download_button(
            label="⬇️ Download Plain Text", 
            data=txt_file, 
            file_name="license_data.txt", 
            mime="text/plain", 
            help="Download data as a simple text file.",
            key=f"download_txt_{unique_key_suffix}" # FIX APPLIED HERE
        )
        
        # 3. Word/DOC Button
        st.download_button(
            label="⬇️ Download Word (.doc)", 
            data=doc_file, 
            file_name="license_data.doc", 
            mime="application/msword", 
            help="Download data as a simple text file, suggesting a Word format.",
            key=f"download_doc_{unique_key_suffix}" # FIX APPLIED HERE
        )

# --- Main App Body ---

st.title("🪪 Myanmar License Extractor (Live & Upload)")
st.write("Choose to upload an image file or use your device's camera to capture the license.")

tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

current_time_suffix = str(time.time()).replace('.', '') # Create a base unique suffix

with tab1:
    st.header("Live Document Capture")
    captured_file = st.camera_input("Place the license clearly in the frame and click 'Take Photo'")
    
    if captured_file is not None:
        # Check if file has been successfully converted to CV2 image
        image_cv = handle_file_to_cv2(captured_file)
        
        if image_cv is not None:
            # FIX: Use a unique key based on the current time and mode
            process_image_and_display(image_cv, f"live_{current_time_suffix}")
        else:
            st.error("Could not read the captured image data. Please try again.")

with tab2:
    st.header("Upload Image File")
    uploaded_file = st.file_uploader("Upload License Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Check if file has been successfully converted to CV2 image
        image_cv = handle_file_to_cv2(uploaded_file)
        
        if image_cv is not None:
            # FIX: Use a unique key based on the current time and mode
            process_image_and_display(image_cv, f"upload_{current_time_suffix}")
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
