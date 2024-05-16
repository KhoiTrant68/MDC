from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch


class Blip2:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_model(self, model_name="Salesforce/blip2-opt-2.7b"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

    def generate_caption(self, image, max_new_tokens=20):
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
