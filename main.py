import streamlit as st
from joblib import load
from PIL import Image 
from util import classify
from skimage.io import imread

st.title('Image classifier')
st.header('Upload an image (jpg, jpeg, bmp)')

file = st.file_uploader('', type = ['jpg', 'jpeg', 'bmp'])

model = load('final_model.joblib')
class_names = ['document', 'image', 'text-embeded image']

if file is not None:
  img_array = imread(file)
  image = Image.open(file)
  st.image(image, use_column_width=True)

  class_name, conf_score = classify(img_array, model, class_names)

  st.write("## Class: {}".format(class_name))
  st.write("### Score (насколько модель уверена в полученном результате):{:.3f}".format(conf_score))