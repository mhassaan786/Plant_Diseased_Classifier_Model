# libs
import torch
from transformers import pipeline

print("SANITY CHECKS GOING ON...")

@staticmethod
def load_clip():
   device = 0 if torch.cuda.is_available() else -1
   
   return pipeline(
     "zero-shot-image-classification",
     model="openai/clip-vit-base-patch32",
     device=device
     )

# Load Hugging Face model
clip = load_clip()

def SanityCheck(image):
#   image = img.convert("RGB")
  check = clip(image, candidate_labels=["leaf", "not_a_leaf"])
  prediction = check[0]["label"]

  if prediction == "leaf":
    return 0
  else:
    return 1