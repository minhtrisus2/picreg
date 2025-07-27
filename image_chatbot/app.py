import streamlit as st
from PIL import Image
import torch
import timm
import json
from urllib.request import urlopen

# --- PHẦN 1: TẢI MÔ HÌNH VÀ NHÃN ---

@st.cache_resource
def load_model():
    """Tải mô hình AI. Chúng ta sẽ dùng ResNet18, một mô hình rất ổn định."""
    # Thay đổi sang mô hình ResNet18
    model = timm.create_model('resnet18', pretrained=True)
    model.eval()
    return model

@st.cache_data
def load_labels():
    """Tải danh sách nhãn từ file JSON.
    Hàm này đã được sửa lỗi triệt để."""
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    return json.load(urlopen(labels_url))

# Khởi tạo mô hình và nhãn
model = load_model()
labels = load_labels()

# --- PHẦN 2: HÀM LOGIC NHẬN DẠNG ---

def recognize_image(image):
    """Xử lý và nhận dạng ảnh."""
    try:
        # Chuẩn bị ảnh để đưa vào mô hình
        data_config = timm.data.resolve_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        tensor = transforms(image).unsqueeze(0)
        
        # Chạy dự đoán
        with torch.no_grad():
            out = model(tensor)
            
        # Xử lý kết quả trả về
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        confidence = top_prob.item() * 100
        label_name = labels[top_catid.item()].replace('_', ' ')
        
        # Trả về chuỗi kết quả
        return f"Đối tượng được xác định là **{label_name.capitalize()}** với độ tin cậy **{confidence:.2f}%**."
    except Exception as e:
        return f"Đã xảy ra lỗi khi xử lý ảnh: {e}"

# --- PHẦN 3: XÂY DỰNG GIAO DIỆN WEB ---

st.set_page_config(layout="wide", page_title="Bot Nhận Dạng Ảnh")

st.title("🤖 Bot Nhận Dạng Hình Ảnh")
st.write("Tải lên một bức ảnh, và AI sẽ cho bạn biết nó nhìn thấy gì.")

uploaded_file = st.file_uploader("Chọn một tệp ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
    
    with col2:
        with st.spinner("Bot đang phân tích..."):
            result = recognize_image(image)
            st.success("Phân tích hoàn tất!")
            st.markdown("### Kết quả:")
            st.markdown(result)
