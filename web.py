import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import tempfile
import os
import time

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="🎨 Media Processing Studio",
    page_icon="🎨",
    layout="wide"
)

# Clean & Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    .main-title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: rgba(255,255,255,0.8);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    .control-group {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .download-btn {
        background: linear-gradient(135deg, #27ae60, #229954) !important;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .image-display {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .video-controls {
        background: #e8f5e8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .processing-status {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .realtime-controls {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🎨 Media Processing Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ประมวลผลภาพและวิดีโอแบบ Professional พร้อม Real-time Camera</p>', unsafe_allow_html=True)

# ฟังก์ชันสำหรับโหลดภาพจาก URL
@st.cache_data
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดภาพจาก URL ได้: {e}")
        return None

# ฟังก์ชันแปลงภาพ
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# ฟังก์ชัน Image Processing
def edge_detection(image, threshold1=50, threshold2=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def blur_image(image, kernel_size=15):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def rotate_image(image, angle=0):
    height, width = image.shape[:2]
    center = (width//2, height//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

def flip_image(image, flip_type=1):
    return cv2.flip(image, flip_type)

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    pil_img = cv2_to_pil(image)
    
    if brightness != 0:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.0 + brightness/100.0)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
    
    return pil_to_cv2(pil_img)

def process_frame(frame, params):
    """ประมวลผล frame เดี่ยวตาม parameters"""
    processed = frame.copy()
    
    if params.get('edge_enable', False):
        processed = edge_detection(processed, params['threshold1'], params['threshold2'])
    
    if params.get('blur_enable', False):
        processed = blur_image(processed, params['blur_intensity'])
    
    if params.get('rotation_enable', False):
        processed = rotate_image(processed, params['angle'])
    
    if params.get('flip_enable', False):
        processed = flip_image(processed, params['flip_type'])
    
    if params.get('bc_enable', False):
        processed = adjust_brightness_contrast(processed, params['brightness'], params['contrast'])
    
    return processed

def create_comparison_histogram(original_image, processed_image):
    """สร้างกราฟเปรียบเทียบภาพก่อน-หลังที่เห็นการเปลี่ยนแปลงชัด"""
    
    # แปลงเป็น grayscale
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # กราฟที่ 1: Grayscale Histogram Comparison
    ax1.hist(gray_original.flatten(), bins=50, alpha=0.7, color='#3498db', 
             label='Original', edgecolor='white', linewidth=0.5)
    ax1.hist(gray_processed.flatten(), bins=50, alpha=0.7, color='#e74c3c', 
             label='Processed', edgecolor='white', linewidth=0.5)
    ax1.set_title('📊 Intensity Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pixel Intensity (0-255)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # คำนวณสถิติ
    mean_orig = np.mean(gray_original)
    mean_proc = np.mean(gray_processed)
    std_orig = np.std(gray_original)
    std_proc = np.std(gray_processed)
    
    ax1.axvline(mean_orig, color='#3498db', linestyle='--', alpha=0.8, label=f'Original Mean: {mean_orig:.1f}')
    ax1.axvline(mean_proc, color='#e74c3c', linestyle='--', alpha=0.8, label=f'Processed Mean: {mean_proc:.1f}')
    
    # เพิ่มข้อมูลสถิติ
    stats_text = f"""Original:
Mean: {mean_orig:.1f}
Std: {std_orig:.1f}

Processed:
Mean: {mean_proc:.1f}
Std: {std_proc:.1f}

Change:
Mean: {mean_proc-mean_orig:+.1f}
Std: {std_proc-std_orig:+.1f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontfamily='monospace')
    
    # กราฟที่ 2: RGB Channels Overlay
    # แยกช่องสี
    r_orig, g_orig, b_orig = cv2.split(original_image)
    r_proc, g_proc, b_proc = cv2.split(processed_image)
    
    # Plot RGB histograms
    bins = np.arange(256)
    
    # Original (เส้นทึบ)
    ax2.plot(bins[:-1], cv2.calcHist([r_orig], [0], None, [255], [0, 255]).flatten(), 
             color='red', alpha=0.8, linewidth=2, label='R Original')
    ax2.plot(bins[:-1], cv2.calcHist([g_orig], [0], None, [255], [0, 255]).flatten(), 
             color='green', alpha=0.8, linewidth=2, label='G Original')
    ax2.plot(bins[:-1], cv2.calcHist([b_orig], [0], None, [255], [0, 255]).flatten(), 
             color='blue', alpha=0.8, linewidth=2, label='B Original')
    
    # Processed (เส้นประ)
    ax2.plot(bins[:-1], cv2.calcHist([r_proc], [0], None, [255], [0, 255]).flatten(), 
             color='red', alpha=0.6, linewidth=2, linestyle='--', label='R Processed')
    ax2.plot(bins[:-1], cv2.calcHist([g_proc], [0], None, [255], [0, 255]).flatten(), 
             color='green', alpha=0.6, linewidth=2, linestyle='--', label='G Processed')
    ax2.plot(bins[:-1], cv2.calcHist([b_proc], [0], None, [255], [0, 255]).flatten(), 
             color='blue', alpha=0.6, linewidth=2, linestyle='--', label='B Processed')
    
    ax2.set_title('🌈 RGB Channel Changes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pixel Intensity (0-255)')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # คำนวณการเปลี่ยนแปลงของแต่ละช่องสี
    r_change = np.mean(r_proc) - np.mean(r_orig)
    g_change = np.mean(g_proc) - np.mean(g_orig)
    b_change = np.mean(b_proc) - np.mean(b_orig)
    
    change_text = f"""Channel Changes:
Red: {r_change:+.1f}
Green: {g_change:+.1f}
Blue: {b_change:+.1f}

Dominant Change:
{['Red', 'Green', 'Blue'][np.argmax(np.abs([r_change, g_change, b_change]))]}"""
    
    ax2.text(0.02, 0.98, change_text, transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📁 เลือกสื่อ</h3>', unsafe_allow_html=True)
    
    media_type = st.selectbox(
        "ประเภทสื่อ:",
        ["🖼️ ภาพ (Image)", "🎬 วิดีโอ (Video)", "📹 Real-time Camera"]
    )
    
    if media_type != "📹 Real-time Camera":
        source_option = st.selectbox(
            "แหล่งที่มา:",
            ["📁 Upload File", "📸 Camera", "🌐 URL"] if media_type == "🖼️ ภาพ (Image)" else ["📁 Upload File"]
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Media Loading
original_image = None
video_file = None
camera_active = False

if media_type == "🖼️ ภาพ (Image)":
    if source_option == "📁 Upload File":
        with st.sidebar:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "เลือกไฟล์ภาพ", 
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
            )
            if uploaded_file:
                original_image = Image.open(uploaded_file)
                st.success("✅ โหลดสำเร็จ!")
            st.markdown('</div>', unsafe_allow_html=True)

    elif source_option == "📸 Camera":
        with st.sidebar:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            camera_image = st.camera_input("📸 ถ่ายภาพ")
            if camera_image:
                original_image = Image.open(camera_image)
                st.success("✅ ถ่ายภาพสำเร็จ!")
            st.markdown('</div>', unsafe_allow_html=True)

    elif source_option == "🌐 URL":
        with st.sidebar:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            image_url = st.text_input("🔗 URL ของภาพ:")
            if image_url:
                original_image = load_image_from_url(image_url)
                if original_image:
                    st.success("✅ โหลดสำเร็จ!")
            st.markdown('</div>', unsafe_allow_html=True)

elif media_type == "🎬 วิดีโอ (Video)":
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "เลือกไฟล์วิดีโอ", 
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        if video_file:
            st.success("✅ โหลดวิดีโอสำเร็จ!")
        st.markdown('</div>', unsafe_allow_html=True)

elif media_type == "📹 Real-time Camera":
    camera_active = True

# Processing Controls
processing_params = {}

with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🛠️ การประมวลผล</h3>', unsafe_allow_html=True)
    
    # Edge Detection
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown("**🔍 Edge Detection**")
    processing_params['edge_enable'] = st.checkbox("เปิดใช้", key="edge")
    if processing_params['edge_enable']:
        col1, col2 = st.columns([2, 1])
        with col1:
            processing_params['threshold1'] = st.slider("Low Threshold", 0, 200, 50)
        with col2:
            processing_params['threshold1'] = st.number_input("", value=processing_params['threshold1'], min_value=0, max_value=200, key="t1_num")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            processing_params['threshold2'] = st.slider("High Threshold", 0, 300, 150)
        with col2:
            processing_params['threshold2'] = st.number_input("", value=processing_params['threshold2'], min_value=0, max_value=300, key="t2_num")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Blur
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown("**🌫️ Blur**")
    processing_params['blur_enable'] = st.checkbox("เปิดใช้", key="blur")
    if processing_params['blur_enable']:
        col1, col2 = st.columns([2, 1])
        with col1:
            processing_params['blur_intensity'] = st.slider("ความแรง", 1, 49, 15, step=2)
        with col2:
            processing_params['blur_intensity'] = st.number_input("", value=processing_params['blur_intensity'], min_value=1, max_value=49, step=2, key="blur_num")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Rotation
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown("**🔄 Rotation**")
    processing_params['rotation_enable'] = st.checkbox("เปิดใช้", key="rotate")
    if processing_params['rotation_enable']:
        col1, col2 = st.columns([2, 1])
        with col1:
            processing_params['angle'] = st.slider("มุม (องศา)", -180, 180, 0)
        with col2:
            processing_params['angle'] = st.number_input("", value=processing_params['angle'], min_value=-180, max_value=180, key="angle_num")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Flip
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown("**🪞 Flip**")
    processing_params['flip_enable'] = st.checkbox("เปิดใช้", key="flip")
    if processing_params['flip_enable']:
        flip_direction = st.selectbox("ทิศทาง:", ["แนวนอน", "แนวตั้ง", "ทั้งสอง"])
        processing_params['flip_type'] = {"แนวนอน": 1, "แนวตั้ง": 0, "ทั้งสอง": -1}[flip_direction]
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Brightness/Contrast
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown("**☀️ แสง/คอนทราสต์**")
    processing_params['bc_enable'] = st.checkbox("เปิดใช้", key="bc")
    if processing_params['bc_enable']:
        col1, col2 = st.columns([2, 1])
        with col1:
            processing_params['brightness'] = st.slider("ความสว่าง", -100, 100, 0)
        with col2:
            processing_params['brightness'] = st.number_input("", value=processing_params['brightness'], min_value=-100, max_value=100, key="bright_num")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            processing_params['contrast'] = st.slider("คอนทราสต์", 0.5, 3.0, 1.0, 0.1)
        with col2:
            processing_params['contrast'] = st.number_input("", value=processing_params['contrast'], min_value=0.5, max_value=3.0, step=0.1, key="contrast_num")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
if camera_active:
    # Real-time Camera
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-header">📹 Real-time Camera Processing</h4>', unsafe_allow_html=True)
    
    # Camera controls
    st.markdown('<div class="realtime-controls">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_camera = st.button("🚀 เริ่มกล้อง", key="start_camera")
    
    with col2:
        capture_frame = st.button("📸 จับภาพ", key="capture_frame")
    
    with col3:
        stop_camera = st.button("⏹️ หยุดกล้อง", key="stop_camera")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Camera display placeholders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📹 Camera Feed**")
        camera_placeholder = st.empty()
    
    with col2:
        st.markdown("**✨ Processed Feed**")
        processed_placeholder = st.empty()
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    
    if start_camera:
        if not st.session_state.camera_running:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.camera_running = True
            st.success("📹 กล้องเริ่มทำงานแล้ว!")
    
    if stop_camera:
        if st.session_state.camera_running and st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.camera_running = False
            st.session_state.cap = None
            st.info("⏹️ หยุดกล้องแล้ว")
    
    # Real-time processing loop
    if st.session_state.camera_running and st.session_state.cap:
        while st.session_state.camera_running:
            ret, frame = st.session_state.cap.read()
            if ret:
                # Process frame
                processed_frame = process_frame(frame, processing_params)
                
                # Convert to display format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frames
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                processed_placeholder.image(processed_rgb, channels="RGB", use_column_width=True)
                
                # Capture frame functionality
                if capture_frame:
                    # Save captured frame for analysis
                    st.session_state.captured_frame = frame.copy()
                    st.session_state.captured_processed = processed_frame.copy()
                    st.success("📸 จับภาพสำเร็จ!")
                
                time.sleep(0.1)  # Small delay to prevent overwhelming
            else:
                st.error("❌ ไม่สามารถอ่านภาพจากกล้องได้")
                break
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show captured frame analysis
    if 'captured_frame' in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-header">📊 การวิเคราะห์ภาพที่จับ</h4>', unsafe_allow_html=True)
        
        if st.button("🔍 วิเคราะห์ภาพที่จับ", key="analyze_captured"):
            processed_captured = process_frame(st.session_state.captured_frame, processing_params)
            fig = create_comparison_histogram(st.session_state.captured_frame, processed_captured)
            st.pyplot(fig)
            plt.close()
            
            # Download captured image
            captured_pil = cv2_to_pil(st.session_state.captured_processed)
            img_buffer = BytesIO()
            captured_pil.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 ดาวน์โหลดภาพที่จับ",
                data=img_bytes,
                file_name="captured_frame.png",
                mime="image/png"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

elif original_image and media_type == "🖼️ ภาพ (Image)":
    cv2_image = pil_to_cv2(original_image)
    processed_image = process_frame(cv2_image, processing_params)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 class="section-header">🖼️ ภาพต้นฉบับ</h4>', unsafe_allow_html=True)
        st.markdown('<div class="image-display">', unsafe_allow_html=True)
        st.image(original_image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h4 class="section-header">✨ ภาพที่ประมวลผลแล้ว</h4>', unsafe_allow_html=True)
        st.markdown('<div class="image-display">', unsafe_allow_html=True)
        processed_pil = cv2_to_pil(processed_image)
        st.image(processed_pil, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download & Analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-header">💾 ดาวน์โหลด</h4>', unsafe_allow_html=True)
        
        img_buffer = BytesIO()
        processed_pil.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        st.download_button(
            label="📥 ดาวน์โหลดภาพ",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png",
            key="download_img"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-header">📊 เปรียบเทียบการเปลี่ยนแปลง</h4>', unsafe_allow_html=True)
        
        if st.button("🔍 สร้างกราฟ", key="analyze"):
            fig = create_comparison_histogram(cv2_image, processed_image)
            st.pyplot(fig)
            plt.close()
            
            # Statistics
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="metric-box"><strong>Original Mean</strong><br>{np.mean(gray):.1f}</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="metric-box"><strong>Processed Mean</strong><br>{np.mean(gray_proc):.1f}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif video_file and media_type == "🎬 วิดีโอ (Video)":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-header">🎬 วิดีโอ Processing</h4>', unsafe_allow_html=True)
    
    # Video preview
    st.video(video_file)
    
    st.markdown('<div class="video-controls">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_video = st.button("🚀 ประมวลผลวิดีโอ", key="process_video")
    
    with col2:
        extract_frame = st.button("📸 สกัดเฟรม", key="extract_frame")
    
    with col3:
        frame_number = st.number_input("เฟรมที่:", min_value=1, value=1, key="frame_num")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if process_video:
        st.markdown('<div class="processing-status">⏳ กำลังประมวลผลวิดีโอ... (อาจใช้เวลาสักครู่)</div>', unsafe_allow_html=True)
        
        # สร้างไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name
        
        # ประมวลผลวิดีโอ
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # สร้างไฟล์เอาต์พุต
        output_path = tempfile.mktemp(suffix='_processed.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ประมวลผลเฟรม
            processed_frame = process_frame(frame, processing_params)
            out.write(processed_frame)
            
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
        
        cap.release()
        out.release()
        
        # แสดงผลลัพธ์
        st.success("✅ ประมวลผลเสร็จสิ้น!")
        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 ดาวน์โหลดวิดีโอ",
                data=f.read(),
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
        
        # ทำความสะอาด
        os.unlink(temp_path)
        os.unlink(output_path)
    
    if extract_frame:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name
        
        cap = cv2.VideoCapture(temp_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        
        if ret:
            st.success(f"✅ สกัดเฟรมที่ {frame_number} สำเร็จ!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**เฟรมต้นฉบับ**")
                frame_pil = cv2_to_pil(frame)
                st.image(frame_pil, use_column_width=True)
            
            with col2:
                st.markdown("**เฟรมที่ประมวลผลแล้ว**")
                processed_frame = process_frame(frame, processing_params)
                processed_pil = cv2_to_pil(processed_frame)
                st.image(processed_pil, use_column_width=True)
                
                # ดาวน์โหลดเฟรม
                img_buffer = BytesIO()
                processed_pil.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                st.download_button(
                    label="📥 ดาวน์โหลดเฟรม",
                    data=img_bytes,
                    file_name=f"frame_{frame_number}.png",
                    mime="image/png"
                )
            
            # วิเคราะห์เฟรม
            if st.button("🔍 วิเคราะห์เฟรม", key="analyze_frame"):
                processed_frame = process_frame(frame, processing_params)
                fig = create_comparison_histogram(frame, processed_frame)
                st.pyplot(fig)
                plt.close()
        
        cap.release()
        os.unlink(temp_path)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome Screen
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🎯 ยินดีต้อนรับ</h3>', unsafe_allow_html=True)
    st.markdown("""
    **Media Processing Studio** เครื่องมือประมวลผลภาพและวิดีโอที่ใช้งานง่าย
    
    **✨ ฟีเจอร์:**
    - 🖼️ ประมวลผลภาพแบบเรียลไทม์
    - 🎬 ประมวลผลวิดีโอ
    - 📹 **Real-time Camera** (ใหม่!)
    - 🔢 ปรับค่าด้วย Slider หรือใส่ตัวเลขโดยตรง
    - 📊 เปรียบเทียบการเปลี่ยนแปลงแบบ Real-time
    - 💾 ดาวน์โหลดผลลัพธ์
    
    **เริ่มต้นใช้งาน:** เลือกสื่อจาก Sidebar ด้านซ้าย
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.7); margin-top: 2rem; font-size: 0.9rem;'>
    🎨 Media Processing Studio | OpenCV + Streamlit + Real-time Processing
</div>
""", unsafe_allow_html=True)