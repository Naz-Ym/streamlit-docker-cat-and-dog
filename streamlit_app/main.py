import io
import os
import streamlit as st
from PIL import Image
from settings import make_detection_img, make_classification_img, classes,\
                     draw_image, model_for_detection, samoyed_model_for_detection, \
                     dict_to_russian, model_for_inference_samoyed, mean_s, std_s, \
                     image_transforms, classes_s



st.markdown('<h1 align="center" style="color:green;">Детектор и классификатор кошки&собаки</h1>',
            unsafe_allow_html=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
col1, col2 = st.columns(2)

activities = ["Детектор", "Классификатор", "Детектор породы собаки Самоед"]
choice = st.sidebar.selectbox("Выберите : ", activities)
main_file_path = os.getcwd()
if choice == 'Детектор' or choice == "Детектор породы собаки Самоед":
    uploaded_files = st.file_uploader("Выберите ваши фото для обработки:",
                                      type=['jpg', 'jpeg', 'png'],
                                      help="Формат должен быть jpg,jpeg,png",
                                      accept_multiple_files=True, )
    n = st.number_input("Выберите количество колонок для отображения фото", 1, 5, 3)
    st.write(f"Обрабатываются {len(uploaded_files)} фото.")
    for i, uploaded_file in enumerate(uploaded_files):
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data)).convert('RGB')
        if choice == "Детектор":
            boxes, cls, probs = make_detection_img(model_for_detection, image)
        else:
            if make_classification_img(image, model=model_for_inference_samoyed, image_transforms=image_transforms, \
                                       mean=mean_s, std=std_s, classes=classes_s) != "samoyed":
                st.image(image)
                st.write("Другое")
                continue
            boxes, cls, probs = make_detection_img(samoyed_model_for_detection, image)
        if len(boxes) == 0:
            st.image(image)
            st.write("Другое")
            continue
        draw_image(image, boxes, i)
        image_path = os.path.join(main_file_path, "image_folders", f"example{i}.jpg")
        image = Image.open(image_path).convert('RGB')
        st.image(image)
        if choice == "Детектор":
            st.write(dict_to_russian[classes[int(cls[0])]], " ", str(round(probs[0]*100, 2)), "%.")
        else:
            st.write("Самоед", " ", str(round(probs[0]*100, 2)), "%.")

elif choice == "Классификатор":
    uploaded_files = st.file_uploader("Выберите ваши фото для обработки:",
                                      type=['jpg', 'jpeg', 'png'],
                                      help="Формат должен быть jpg,jpeg,png",
                                      accept_multiple_files=True, )
    n = st.number_input("Выберите количество колонок для отображения фото", 1, 5, 3)
    st.write(f"Обрабатываются {len(uploaded_files)} фото.")
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data)).convert('RGB')
        st.image(image)
        st.write(make_classification_img(image))

