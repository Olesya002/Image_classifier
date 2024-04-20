import gradio as gr
from joblib import load
import cv2
import torch
import piqa
import numpy as np

model = load('final_model.joblib')
fid_metric = piqa.FID()
class_names = ['document', 'image', 'text-embeded image']

def classifier(image):
    #img_array = imread(image, as_gray=False)
    if len(image.shape) != 3 or image.shape[2] != 3:
      image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = (250, 250)
    resized_img = cv2.resize(image, size)
    #bright_img = cv2.add(resized_img, 50)
    img_tensor = torch.tensor(resized_img).permute(2, 0, 1)[None, ...] / 255
    img_feats = fid_metric.features(img_tensor).reshape(-1).reshape(1,-1)
    prediction = model.predict_proba(img_feats)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score
    
#define gradio interface and other parameters
app =  gr.Interface(
   fn = classifier, 
   inputs = gr.Image(label='Загрузите изображение (jpg, jpeg, bmp):'), 
   outputs =  [gr.components.Textbox(label = 'Класс изображения:'), gr.components.Textbox(label = 'Точность результата:')],
   allow_flagging=False)

app.launch(share = True)