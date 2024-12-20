import os
import json
import argparse
import pandas as pd
from typing import Union
from PIL import Image
import mimetypes

import torch

import transformers

from .configs.lora_config import openflamingo_tuning_config

from .mllm.src.factory import create_model_and_transforms

from huggingface_hub import hf_hub_download

from peft import (
    get_peft_model,
    LoraConfig,
    get_peft_model_state_dict,
    PeftConfig,
    PeftModel
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image Handling Functions -------------------
def get_image(file_path: str) -> Union[Image.Image, list]:
    content_type = get_content_type(file_path)
    if "image" in content_type:
        return Image.open(file_path)
    else:
        raise ValueError("Invalid content type. Expected image.")

def load_pretrained_model():
    peft_config, peft_model_id = None, None
    peft_config = LoraConfig(**openflamingo_tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14-336",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b", # anas-awadalla/mpt-7b
        tokenizer_path="anas-awadalla/mpt-7b",  # anas-awadalla/mpt-7b
        cross_attn_every_n_layers=4,
        use_peft=True,
        peft_config=peft_config,
    )

    checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.half().cuda()

    return model, image_processor, tokenizer

def get_model_inputs(image_path, instruction, model, image_processor, tokenizer):
    image = get_image(image_path)
    # Process image and expand dimensions to fit the expected shape: (b, T_img, F, C, H, W)
    # Here, we assume batch size of 1, 1 image sequence, 1 frame per sequence
    vision_x = image_processor(image).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Adding dimensions
    vision_x = vision_x.expand(-1, 1, 1, -1, -1, -1)  # Adjust dimensions if necessary
    prompt = f"USER: <image> {instruction} GPT:<answer>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    return vision_x, inputs

if __name__ == "__main__":
    image_path = "./playground/images/sample.jpg"
    instruction = "Are any moving bicycles visible?"

    model, image_processor, tokenizer = load_pretrained_model()
    vision_x, inputs = get_model_inputs(image_path, instruction, model, image_processor, tokenizer)

    generation_kwargs = {
        'max_new_tokens': 512,
        'temperature': 1,
        'top_k': 0,
        'top_p': 1,
        'no_repeat_ngram_size': 3,
        'length_penalty': 1,
        'do_sample': False,
        'early_stopping': True
    }

    generated_tokens = model.generate(
        vision_x=vision_x.half().cuda(),  # Half precision
        lang_x=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        num_beams=3,
        **generation_kwargs,
    )

    generated_text = tokenizer.batch_decode(generated_tokens.cpu().numpy())

    print(
        f"Dolphin output:\n\n{generated_text}"
    )
