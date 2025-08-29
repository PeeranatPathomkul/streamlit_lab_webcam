# streamlit_lab_webcam
6610110214 Peeranat Pathomkul
# Media Processing Studio

เครื่องมือประมวลผล Image และ Video แบบ Real-time สร้างด้วย Streamlit และ OpenCV

## คุณสมบัติหลัก

### แหล่งข้อมูล Media
- **Upload File**: อัพโหลดไฟล์ภาพและวิดีโอจากเครื่องคอมพิวเตอร์
- **Camera Capture**: ถ่ายภาพโดยใช้ webcam ของเครื่อง
- **URL Input**: โหลดภาพจาก internet ผ่าน URL
- **Real-time Camera**: ประมวลผลภาพจากกล้องแบบ real-time

### การประมวลผลภาพ
- **Edge Detection**: ตรวจจับขอบด้วยอัลกอริทึม Canny
- **Blur Effect**: เบลอภาพด้วย Gaussian Blur
- **Image Rotation**: หมุนภาพตามมุมที่กำหนด
- **Image Flipping**: สะท้อนภาพแนวนอน แนวตั้ง หรือทั้งสองทิศทาง
- **Brightness & Contrast**: ปรับความสว่างและความคมชัดของภาพ

### การควบคุม Parameters
- **Dual Input System**: ปรับค่าได้ทั้งแบบ slider และการใส่ตัวเลขโดยตรง
- **Multi-effect Processing**: ใช้งาน effect หลายตัวพร้อมกันได้
- **Real-time Preview**: ดูผลลัพธ์ทันทีเมื่อเปลี่ยนค่า

### การวิเคราะห์ข้อมูล
- **Histogram Comparison**: เปรียบเทียบการกระจายของ pixel intensity
- **RGB Channel Analysis**: วิเคราะห์การเปลี่ยนแปลงในแต่ละช่องสี
- **Statistical Metrics**: แสดงค่าทางสถิติเช่น Mean และ Standard Deviation

## การติดตั้งและใช้งาน

### Dependencies ที่จำเป็น
```bash
pip install streamlit opencv-python pillow matplotlib requests numpy
```

### การรันโปรแกรม
```bash
streamlit run app.py
```

## วิธีใช้งาน

### 1. เลือก Media Type
- เลือกประเภท media ที่ต้องการประมวลผล (Image, Video, Real-time Camera)

### 2. โหลดข้อมูล
- **สำหรับภาพ**: เลือกจาก Upload, Camera หรือ URL
- **สำหรับวิดีโอ**: อัพโหลดไฟล์วิดีโอ
- **สำหรับ Real-time**: กดปุ่ม "เริ่มกล้อง"

### 3. ตั้งค่า Processing Parameters
- เปิดใช้งาน effect ที่ต้องการในแต่ละหมวด
- ปรับค่าต่างๆ ด้วย slider หรือใส่ตัวเลขโดยตรง

### 4. ดูผลลัพธ์และวิเคราะห์
- ดูภาพหรือวิดีโอที่ประมวลผลแล้วแบบ real-time
- กดปุ่ม "สร้างกราฟ" เพื่อดูการวิเคราะห์
- ดาวน์โหลดผลลัพธ์ได้

## รายละเอียด Effect แต่ละตัว

### Edge Detection
- **Low Threshold**: ค่าขีดจำกัดล่างสำหรับการตรวจจับขอบ (0-200)
- **High Threshold**: ค่าขีดจำกัดบนสำหรับการตรวจจับขอบ (0-300)

### Blur Effect
- **Intensity**: ความแรงของการเบลอ (1-49, เฉพาะตัวเลขคี่)

### Rotation
- **Angle**: มุมการหมุนในหน่วยองศา (-180 ถึง +180)

### Flipping
- **แนวนอน**: สะท้อนซ้าย-ขวา
- **แนวตั้ง**: สะท้อนบน-ล่าง
- **ทั้งสอง**: สะท้อนทั้งสองทิศทาง

### Brightness & Contrast
- **Brightness**: ความสว่าง (-100 ถึง +100)
- **Contrast**: ความคมชัด (0.5 ถึง 3.0)

## คุณสมบัติ Video Processing

### การประมวลผลวิดีโอทั้งไฟล์
- อัพโหลดวิดีโอรูปแบบ MP4, AVI, MOV, MKV
- ประมวลผลทุกเฟรมด้วย effect ที่เลือก
- แสดง progress bar ระหว่างการประมวลผล
- ดาวน์โหลดวิดีโอที่ประมวลผลแล้ว

### การสกัดเฟรม
- เลือกเฟรมที่ต้องการจากวิดีโอ
- ดูภาพก่อนและหลังการประมวลผล
- วิเคราะห์เฟรมด้วยกราฟ
- ดาวน์โหลดเฟรมเป็นไฟล์ภาพ

## Real-time Camera Features

### การทำงานแบบ Real-time
- แสดงภาพจากกล้องแบบสด
- ประมวลผลและแสดงผลทันที
- ปรับ parameter และเห็นผลทันที

### การจับภาพ
- กดปุ่ม "จับภาพ" เพื่อบันทึกเฟรมปัจจุบัน
- วิเคราะห์ภาพที่จับได้ด้วยกราฟ
- ดาวน์โหลดภาพที่ประมวลผลแล้ว

## การวิเคราะห์ข้อมูลด้วยกราฟ

### Intensity Distribution Comparison
- เปรียบเทียบการกระจายของ pixel intensity ก่อนและหลังการประมวลผล
- แสดงค่าเฉลี่ย (Mean) และความเบี่ยงเบนมาตรฐาน (Standard Deviation)
- แสดงการเปลี่ยนแปลงเป็นตัวเลข

### RGB Channel Analysis
- วิเคราะห์การเปลี่ยนแปลงในแต่ละช่องสี (Red, Green, Blue)
- เปรียบเทียบ histogram ของแต่ละสีก่อนและหลัง
- บอกช่องสีที่มีการเปลี่ยนแปลงมากที่สุด

## เทคโนโลยีที่ใช้

- **Streamlit**: สร้าง Web Interface
- **OpenCV**: ประมวลผลภาพและวิดีโอ
- **PIL (Pillow)**: จัดการไฟล์ภาพ
- **Matplotlib**: สร้างกราฟและการวิเคราะห์
- **NumPy**: คำนวณทางคณิตศาสตร์

## ข้อกำหนดระบบ

- Python 3.7 หรือสูงกว่า
- Webcam สำหรับ Real-time Camera mode
