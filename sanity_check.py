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

def SanityCheck(image, threshold=0.40):

    labels=[
        "a close-up photograph of a plant leaf", 
        "a photograph of something that is not a plant leaf"
        ]
  
    check = clip(image, candidate_labels=labels)

    leaf_score = next(
        r["score"] for r in check
        if r["label"] == labels[0]
    )

    if leaf_score >= threshold:
        return 0

    return 1