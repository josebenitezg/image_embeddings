from transformers import ViTForImageClassification

# import fine-tuned version of model from Hugging Face hub (if needed)
model_id = 'LaCarnevali/vit-cifar10'
model = ViTForImageClassification.from_pretrained(model_id)