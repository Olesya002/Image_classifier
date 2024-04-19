import gradio as gr
from joblib import load
import cv2
import torch
import piqa

clf = load('model_svc.joblib')
fid_metric = piqa.FID()
category = {0: 'document', 1: 'image', 2: 'text-embeded'}

def user_greeting(image):
    #img_array = imread(image, as_gray=False)
    if len(image.shape) != 3 or image.shape[2] != 3:
      image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = (250, 250)
    resized_img = cv2.resize(image, size)
    bright_img = cv2.add(resized_img, 50)
    img_tensor = torch.tensor(bright_img).permute(2, 0, 1)[None, ...] / 255
    img_feats = fid_metric.features(img_tensor).reshape(-1).reshape(1,-1)
    res = clf.predict(img_feats)
    return category[res[0]]
    
#define gradio interface and other parameters
app =  gr.Interface(
   fn = user_greeting, 
   inputs = gr.Image(label='Загрузите изображение (jpg, fpeg, bmp):'), 
   outputs = gr.components.Textbox(label = 'Класс изображения:'),
   allow_flagging=False)

app.launch(share = True)