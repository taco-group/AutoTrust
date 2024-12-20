"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import sys

sys.path.append("..")

import json
import os
import random

import torch
import numpy as np
from PIL import Image
from transformers import LlamaTokenizer

from dataset_utils.vqa_dataset import VQADataset
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import  collate_fn

QUESTIONS = [
        "Describe the following image in detail.",
        "Provide a detailed description of the given image.",
        "Give an elaborate explanation of the image you see.",
        "Share a comprehensive rundown of the presented image.",
        "Offer a thorough analysis of the image.",
        "Explain the various aspects of the image before you.",
        "Clarify the contents of the displayed image with great detail.",
        "Characterize the image using a well-detailed description.",
        "Break down the elements of the image in a detailed manner.",
        "Walk through the important details of the image.",
        "Portray the image with a rich, descriptive narrative.",
        "Narrate the contents of the image with precision.",
        "Analyze the image in a comprehensive and detailed manner.",
        "Illustrate the image through a descriptive explanation.",
        "Examine the image closely and share its details.",
        "Write an exhaustive depiction of the given image.",
]


class COCOSICDataset(VQADataset):
    def __init__(self, tokenizer, vis_processor=None, vis_root=None, ann_paths=[], **kwargs):
        super().__init__(tokenizer, vis_processor, vis_root, ann_paths, **kwargs)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """


    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann['coco_image'])
        image = Image.open(image_path).convert("RGB")
        image = [self.vis_processor(image)]

        region_images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(region_images), self.max_num_images))
        images_tensors = region_images[keep_ixs]

        # pad to 5 images
        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size, self.vis_processor.image_size),
                dtype=torch.float
            )
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

        image = images_tensors.unsqueeze(1)

        return image

    def process_text(self, ann):
        image_caption = ann["image_caption"]
        instruction = random.choice(QUESTIONS)
        src_text = f"<image> User: {instruction} GPT:<answer> {image_caption} <|endofchunk|>"

        return dict(src_text=src_text)

    def tokenize(self, text):
        res = self.tokenizer(text["src_text"], return_tensors="pt", padding="do_not_pad", truncation=True,
                             max_length=self.max_seq_length, add_special_tokens=False)
        res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        return res

    def collater(self, samples):

        return collate_fn(samples,
                          pad_idx=self.tokenizer.pad_token_id,
                          eos_idx=self.tokenizer.eos_token_id,
                          left_pad=self.tokenizer.padding_side == "left"
                          )


from transformers import LlamaTokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader
import  matplotlib.pyplot as plt


if __name__ == "__main__":

    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]
    })

    image_processor = DefaultTransform()

    dataset = COCOSICDataset(
        tokenizer=tokenizer,
        vis_processor=image_processor,
        vis_root="/home/yingzi/vlm/workspace/dataset/COCO/train2017",
        ann_paths=
        ["/home/yingzi/vlm/workspace/dataset/Captions/sic/captions_fliter.jsonl",
         ]
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        #sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    for batch in train_dataloader:
        batch = batch['net_input']
        for k, v in batch.items():
            if isinstance(v, list):
                print(k, len(v))
                continue
            print(k, v.shape)

        images = batch['image']
        image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.savefig("/home/yingzi/vlm/examples/sic.png")

        print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
        break