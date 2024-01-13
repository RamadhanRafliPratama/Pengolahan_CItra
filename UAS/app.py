import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

def apply_dilation_erosion(image, operation_type):
    # Konversi ke citra grayscale jika belum
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tentukan kernel untuk dilasi atau erosi
    kernel = np.ones((5, 5), np.uint8)

    # Lakukan operasi dilasi atau erosi
    if operation_type == "dilation":
        result_image = cv2.dilate(image, kernel, iterations=1)
    elif operation_type == "erosion":
        result_image = cv2.erode(image, kernel, iterations=1)
    else:
        result_image = image

    return result_image

def apply_emboss(image):
    # Matriks kernel untuk filter emboss
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])

    # Melakukan konvolusi dengan kernel emboss
    emboss_image = cv2.filter2D(image, -1, kernel)

    # Normalisasi hasil konvolusi untuk memastikan nilai piksel berada dalam rentang 0-255
    emboss_image = cv2.normalize(emboss_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert ke tipe data uint8
    emboss_image = emboss_image.astype(np.uint8)

    return emboss_image

def apply_resize_rotate(image, resize_factor, rotation_angle):
    resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
    rotated_image = Image.fromarray(resized_image)
    rotated_image = rotated_image.rotate(rotation_angle, resample=Image.BICUBIC)
    return np.array(rotated_image)

def apply_filter(image, filter_type):
    if filter_type == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif filter_type == "lowpass":
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "highpass":
        lowpass = cv2.GaussianBlur(image, (5, 5), 0)
        return image - lowpass
    elif filter_type == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "emboss":
        return apply_emboss(image)
    elif filter_type == "inverse":
        return cv2.bitwise_not(image)
    else:
        return image

def apply_segmentation(image, segmentation_type, seed_point=None, threshold=None):
    if segmentation_type == "edge_detection":
        return cv2.Canny(image, 100, 200)
    elif segmentation_type == "regional_growing":
        return apply_regional_growing(image, seed_point, threshold)
    else:
        return image

def apply_regional_growing(image, seed_point, threshold):
    if seed_point is None or threshold is None:
        return image

    # Salin gambar agar tidak merusak gambar asli
    segmented_image = image.copy()

    # Konversi ke citra grayscale jika belum
    if len(segmented_image.shape) > 2:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Tentukan kondisi berhenti regional growing
    condition = np.zeros_like(segmented_image, dtype=np.uint8)

    # Atur nilai piksel pada seed point sebagai kondisi awal
    condition[seed_point[1], seed_point[0]] = 1

    # Lakukan iterasi hingga kondisi tidak berubah
    while True:
        last_condition = condition.copy()

        # Perbarui kondisi menggunakan operasi morphological dilate
        condition = cv2.dilate(condition, np.ones((3, 3), np.uint8), iterations=1)

        # Tentukan piksel yang memenuhi kriteria untuk dilatasi lebih lanjut
        new_pixels = np.logical_and(segmented_image > 0, np.logical_and(segmented_image <= threshold, condition > 0))

        # Tambahkan piksel baru ke kondisi
        condition[new_pixels] = 1

        # Hentikan iterasi jika kondisi tidak berubah
        if np.array_equal(last_condition, condition):
            break

    # Gunakan kondisi sebagai mask untuk memperoleh hasil akhir
    result = np.zeros_like(segmented_image)
    result[condition > 0] = segmented_image[condition > 0]

    return result

def main():
    st.title("Aplikasi Pengolahan Gambar")

    # Elemen input untuk mengunggah gambar
    uploaded_image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    # Jika gambar berhasil diunggah, tampilkan gambar tersebut
    if uploaded_image is not None:
        original_image = np.array(Image.open(uploaded_image))

        # Periksa dan sesuaikan ukuran gambar
        max_image_size = 1000  # Sesuaikan dengan ukuran maksimum yang diizinkan
        if original_image.shape[0] > max_image_size or original_image.shape[1] > max_image_size:
            st.warning(f"Ukuran gambar terlalu besar. Otomatis menyesuaikan ukuran ke {max_image_size}x{max_image_size}")
            scale_factor = max_image_size / max(original_image.shape[0], original_image.shape[1])
            original_image = cv2.resize(original_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # Resize gambar untuk preview
        preview_image = original_image.copy()

        st.image(preview_image, caption="Gambar yang telah diunggah (Preview)", use_column_width=False)

        # Kolom "Edit"
        st.sidebar.header("Edit")
        resize_factor = st.sidebar.slider("Resize", 0.1, 2.0, 1.0, step=0.1)
        rotation_angle = st.sidebar.slider("Rotate", -180, 180, 0)

        # Kolom "Filter"
        st.sidebar.header("Filter")
        filter_type = st.sidebar.selectbox("Pilih filter", ["none", "grayscale", "HSV", "lowpass", "highpass", "gaussian", "emboss", "inverse"])

        # Kolom "Metode Segmentasi"
        st.sidebar.header("Metode Segmentasi")
        segmentation_type = st.sidebar.selectbox("Pilih metode segmentasi", ["none", "edge_detection", "regional_growing"])

        # Inputan Seed Point dan Threshold untuk Regional Growing
        if segmentation_type == "regional_growing":
            st.sidebar.header("Regional Growing")
            seed_x = st.sidebar.number_input("Seed Point X", value=0, step=1)
            seed_y = st.sidebar.number_input("Seed Point Y", value=0, step=1)
            threshold = st.sidebar.slider("Threshold", 1, 255, 128)

            seed_point = (seed_x, seed_y)
        else:
            seed_point = None
            threshold = None

        # Operasi Dilasi atau Erosi
        st.sidebar.header("Operasi Morfologi")
        morphological_operation = st.sidebar.radio("Pilih operasi morfologi", ["none", "dilation", "erosion"])

        # Memproses gambar sesuai dengan opsi yang dipilih tanpa menggunakan tombol submit
        processed_image = original_image.copy()
        processed_image = apply_resize_rotate(processed_image, resize_factor, rotation_angle)
        processed_image = apply_filter(processed_image, filter_type)
        processed_image = apply_segmentation(processed_image, segmentation_type, seed_point, threshold)

        # Operasi Morfologi: Dilasi atau Erosi
        if morphological_operation != "none":
            processed_image = apply_dilation_erosion(processed_image, morphological_operation)

        # Menampilkan gambar hasil pemrosesan
        st.image(processed_image, caption="Gambar hasil pemrosesan", use_column_width=False)

        # Kolom "Save As"
        st.sidebar.header("Save As")
        save_as_type = st.sidebar.selectbox("Pilih format penyimpanan", [".jpg", ".png", ".pdf"])

        # Tombol "Save" untuk menyimpan gambar
        if st.sidebar.button("Save"):
            # Menyimpan gambar sesuai dengan format yang dipilih
            output_path = f"output{save_as_type}"
            cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            st.success(f"Gambar berhasil disimpan sebagai {output_path}")

if __name__ == "__main__":
    main()
