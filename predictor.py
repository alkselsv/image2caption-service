import torch
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor:
  def __init__(self):
    self.model = Model.load_model().to(device)
    self.feature_extractor = Model.load_feature_extractor()
    self.tokenizer = Model.load_tokenizer()
    self.gen_kwargs = {"max_length": 16, "num_beams": 4}

  def predict_caption(self, image):
    pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values 
    pixel_values = pixel_values.to(device)
    output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
    preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds