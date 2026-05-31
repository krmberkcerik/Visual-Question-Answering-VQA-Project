import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

image = Image.open("test_image.jpg").convert("RGB")
question = "What color is the circle in the center?"

inputs = processor(image, question, return_tensors="pt").to(device)
output_ids = model.generate(**inputs)
answer = processor.decode(output_ids[0], skip_special_tokens=True)

print(f"\nSoru: {question}")
print(f"Cevap: {answer}")
