import cv2
import torch
import piqa
import numpy as np

def classify(img_array, model, class_names):
  fid_metric = piqa.FID()
  if len(img_array.shape) != 3 or img_array.shape[2] != 3:
      img_array =  cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
  size = (250, 250)
  resized_img = cv2.resize(img_array, size)
  bright_img = cv2.add(resized_img, 50)
  img_tensor = torch.tensor(bright_img).permute(2, 0, 1)[None, ...] / 255
  img_feats = fid_metric.features(img_tensor).reshape(-1).reshape(1,-1)
  prediction = model.predict_proba(img_feats)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]
  return class_name, confidence_score