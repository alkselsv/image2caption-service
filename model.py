from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class Model:

  def __init__(self):
    pass

  def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("./model")
    return model

  def load_feature_extractor():
    feature_extractor = ViTImageProcessor.from_pretrained("./model")
    return feature_extractor

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    return tokenizer