import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO


def apply_resize_rotate(image, resize_factor, rotation_angle):
    resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
    rotated_image = Image.fromarray(resized_image)
    rotated_image = rotated_image.rotate(rotation_angle, resample=Image.BICUBIC)
    return np.array(rotated_image)

def apply_filter(image, filter_type):
    if filter_type == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif filter_type == "Lowpass":
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Highpass":
        lowpass = cv2.GaussianBlur(image, (5, 5), 0)
        return image - lowpass
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "Emboss":
        return apply_emboss(image)
    elif filter_type == "Inverse":
        return cv2.bitwise_not(image)
    else:
        return image

def apply_emboss(image):
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])

    emboss_image = cv2.filter2D(image, -1, kernel)
    emboss_image = cv2.normalize(emboss_image, None, 0, 255, cv2.NORM_MINMAX)
    emboss_image = emboss_image.astype(np.uint8)

    return emboss_image

def apply_dilation_erosion(image, operation_type):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)

    if operation_type == "Dilation":
        result_image = cv2.dilate(image, kernel, iterations=1)
    elif operation_type == "Erosion":
        result_image = cv2.erode(image, kernel, iterations=1)
    elif operation_type == "Countour":
        return apply_contour(image)
    else:
        result_image = image

    return result_image

def apply_contour(image):
    # Ubah citra menjadi citra grayscale jika belum
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ambil kontur menggunakan metode yang diperbaiki
    img_binary = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)[1]
    img_binary = ~img_binary
    (contours, _) = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = len(contours)

    # Membuat salinan citra untuk menggambar kontur tanpa memengaruhi citra asli
    image_with_contour = image.copy()

    # Gambar kontur pada citra hasil pemrosesan
    cv2.drawContours(image_with_contour, contours, -1, (0, 255, 0), 2)

    # Tambahkan teks hanya jika terdapat kontur
    if count > 0:
        cv2.putText(image_with_contour, f"Objek: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    font_scale = 0.1
    return image_with_contour


def apply_segmentation(image, segmentation_type):
    if segmentation_type == "Edge Detection":
        return cv2.Canny(image, 100, 200)
    else:
        return image

def save_as_pdf(image):
    pdf_output = BytesIO()
    pdf_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pdf_image = ImageOps.exif_transpose(pdf_image)
    pdf_image = pdf_image.convert("L")
    pdf_image.save(pdf_output, format='pdf')

    return pdf_output.getvalue()

def main():
    st.title("Web Pengolahan Citra Sederhana")

    uploaded_image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        original_image = np.array(Image.open(uploaded_image))

        max_image_size = 1000
        if original_image.shape[0] > max_image_size or original_image.shape[1] > max_image_size:
            st.warning(f"Ukuran gambar terlalu besar. Otomatis menyesuaikan ukuran ke {max_image_size}x{max_image_size}")
            scale_factor = max_image_size / max(original_image.shape[0], original_image.shape[1])
            original_image = cv2.resize(original_image, (0, 0), fx=scale_factor, fy=scale_factor)

        preview_image = original_image.copy()
        st.image(preview_image, caption="Gambar yang telah diunggah (Preview)", use_column_width=False)

        st.sidebar.header("Edit")
        resize_factor = st.sidebar.slider("Resize", 0.1, 2.0, 1.0, step=0.1)
        rotation_angle = st.sidebar.slider("Rotate", -180, 180, 0)

        st.sidebar.header("Filter")
        filter_type = st.sidebar.selectbox("Pilih filter", ["None", "Grayscale", "HSV", "Lowpass", "Highpass", "Gaussian", "Emboss", "Inverse"])

        st.sidebar.header("Metode Segmentasi")
        segmentation_type = st.sidebar.selectbox("Pilih metode segmentasi", ["None", "Edge Detection"])

        st.sidebar.header("Operasi Morfologi")
        morphological_operation = st.sidebar.radio("Pilih operasi morfologi", ["None", "Dilation", "Erosion", "Countour"])

        processed_image = original_image.copy()
        processed_image = apply_resize_rotate(processed_image, resize_factor, rotation_angle)
        processed_image = apply_filter(processed_image, filter_type)
        processed_image = apply_segmentation(processed_image, segmentation_type)

        if morphological_operation != "None":
            processed_image = apply_dilation_erosion(processed_image, morphological_operation)

        st.image(processed_image, caption="Gambar hasil pemrosesan", use_column_width=False)

        st.sidebar.header("Download As")
        download_as_type = st.sidebar.selectbox("Pilih format download", [".jpg", ".png", ".pdf"])

        if st.sidebar.button("Download"):
            output_path = f"output{download_as_type}"
            
            if download_as_type == ".pdf":
                pdf_bytes = save_as_pdf(processed_image)
                st.success(f"Gambar berhasil disimpan sebagai {output_path}")
                st.download_button(label="Download", data=pdf_bytes, file_name=output_path, mime="application/pdf")
            else:
                bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, bgr_image)
                st.success(f"Gambar berhasil disimpan sebagai {output_path}")
                with open(output_path, "rb") as f:
                    file_content = f.read()
                st.download_button(label="Download", data=file_content, file_name=output_path, mime=f"image/{download_as_type[1:]}")

if __name__ == "__main__":
    main()
