<p align="center" width="60%">
<img src="assets/logo.png" alt="OpenEMMA" style="width: 35%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/Project%20Page-8A2BE2)](https://taco-group.github.io/AutoTrust/)
![Code License](https://img.shields.io/badge/Code%20License-Apache%202.0-brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-2412.15206-b31b1b.svg)](https://arxiv.org/abs/2412.15206)

</div>



# AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving

**AutoTrust** is a groundbreaking benchmark designed to assess the trustworthiness of DriveVLMs. This work aims to enhance public safety by ensuring DriveVLMs operate reliably across critical dimensions.

### Key Highlights

- **Comprehensive Evaluation:** Examines DriveVLMs across five pillars: Trustfulness, Safety, Robustness, Privacy, and Fairness, over 10k unique scenes and 18k Q&A pairs constructed to evaluate diverse driving scenarios.

- **Generalists Shine:** Surprisingly, generalist models like GPT-4o-mini outperform specialized models in trustworthiness.

- **Adversarial Vulnerabilities:** Identifies significant weaknesses in privacy protection, bias, and resilience to attacks, demanding urgent attention.

<div align="center">
  <img src="assets/autotrust-teaser.png" alt="EMMA diagram" width="800"/>
  <p><em>Figure 1. We present AUTOTRUST, a comprehensive benchmark for assessing the trustworthiness of large vision language models for autonomous driving (i.e., DriveVLMs), covering five key dimensions: Trustfulness, Safety, Robustness, Privacy, and Fairness.</em></p>
</div>

### News
- **[2024/12/19]** 🔥We released **AutoTrust**, a groundbreaking benchmark designed to assess the trustworthiness of DriveVLMs.. Explore our [paper](https://arxiv.org/abs/2412.15206) and [website](https://taco-group.github.io/AutoTrust/) for more details.

### Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Contact](#contact)
- [Citation](#citation)

## Installation
```
conda env create -f environment.yml
conda activate trust
```
You may need to update your PyTorch version to be compatible with your specific CUDA version and environment. After updating PyTorch, if you encounter issues with any packages, try upgrading them as well.

Example:
```
pip --upgrade numpy pycocotools
```

If you encounter the issus about importing `_expand_mask` and `_make_causal_mask`, you just need to paste the following into `home/[user-name]/ananconda3/envs/trust/lib/python3.10/site-packages/transformers/models/bloom/modeling_bloom.py` and `home/[user-name]/ananconda3/envs/trust/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py`

```
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
        mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
```


## Dataset
Download the raw image files from the following sources:
- [Nuscenes](https://www.nuscenes.org/nuscenes#download)
- [DriveLM-Nuscenes](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/drivelm_nus_imgs_val.zip)
- [LingoQA](https://drive.google.com/file/d/1U5tBTZjw0rH9ID1JuuWsOmIp2HNvQvQo/view?usp=drive_link)
- [CoVLA](https://drive.google.com/drive/folders/17kMdqza6Q6ueWyDpxSIVEC8ZW3lTxtVI?usp=drive_link)

Once downloaded, unzip the files into your designated directory `[path-to-image]`.

## Model
### Llava
If you can get access to the huggingface, please change the `--model-path` in `run.sh` to `liuhaotian/llava-v1.6-mistral-7b`. 

Otherwise, you need first download the [llava-v1.6-mistral-7b](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b), and then download the [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336). Lastly, change the ` "mm_vision_tower": "openai/clip-vit-large-patch14-336"` to ` "mm_vision_tower": "<your-clip-dir>"` in `<your-llava-1.6-dir>/config.json`.

### EM_VLM4AD
You can follow the instruction of [EM_VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD) by putting the [model weight](https://drive.google.com/drive/folders/12bHyRTpWWxIJ2pb0WWzfX5mMdkNHKMVP) under `./EM_VLM4AD/multi_frame_results/T5-Large`. You can then modify the `run.sh` file to run inference on the EM_VLM4AD model.

### DriveLM-Challenge(LLaMA-Adapter-V2)
* Obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure
  ```
  /path/to/llama_model_weights
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  └── tokenizer.model
  ```

* Obtain the [Model Checkpoint](https://huggingface.co/reproduce/drivelm-llama-adapter-v2-ckpt)
* You can then modify the `run.sh` file to run inference on the DriveLM-Challenge model.

## Usage
After setting up the environment, you can begin evaluating the model's performance in the specific trustworthiness dimension by following these instructions. Currently, we support the following models: `GPT-4o-mini`, `LLaVA-1.6-Mistral-7B`, `DriveLM-Agent`, `DriveLM-Challenge`, and `Dolphins`, and `EM_VLM4AD`. To use a specific model, simply pass `gpt`, `liuhaotian/llava-v1.6-mistral-7b`, `drivelm-agent`, `LLaMA`, `Dolphins`, `EM_VLM4AD` as the argument to `--model-path`.

### Trustfulnesss

#### Factuality
For evaluating factuality of the `llava-v1.6-mistral-7b` on `DriveLM-NuScenes`:
```
python main.py \
    --task factuality \
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --image-folder [images-folder-of-nuscence] \
    --question-file ./qa-pairs/trustfulness/drivelm.json \
    --answers-file [factuality-output-dir]
```

#### Uncertainty
For evaluating uncertainty of the `llava-v1.6-mistral-7b` on `DriveLM-NuScenes`:
```
python main.py \
    --task uncertainty \
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --image-folder [images-folder-of-nuscence] \
    --question-file [factuality-output-dir] \
    --answers-file [uncertainty-output-dir]
```
### Safety
For evaluating the performance of the `llava-v1.6-mistral-7b` on `DriveLM-NuScenes` under `black_box_attack`, `misinformation`,  `misinstruction`, or `white_box_attack`:
```
python main.py \
    --task [black_box_attack/misinformation/misinstruction/white_box_attack] \
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --dataset DriveLM \
    --image-folder [images-folder-of-nuscence] \
    --question-file ./qa-pairs/trustfulness/drivelm.json \
    --answers-file [safety-output-dir] \
    --gt-file ./qa-pairs/trustfulness/drivelm.json
```

### Robustness
For evaluating the robutsness of the `llava-v1.6-mistral-7b` on `DriveLM-NuScenes` in the linguistic domain:
```
python main.py \
    --task [robustness_acc/robustness_abs] \
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --image-folder [images-folder-of-nuscence] \
    --question-file ./qa-pairs/robustness/[qa-file] \
    --answers-file [robutsness-output-dir]
```

### Privacy
For evaluating the performance of the `llava-v1.6-mistral-7b` on `identity_individual`, `identity_vehicle`, or `location`:
```
python main.py \
    --task  [identity_individual/identity_vehicle/location]\
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --image-folder [images-folder-of-nuscence] \
    --question-file ./qa-pairs/private/[qa-file] \
    --answers-file [privacy-output-dir]
```

### Fairness

#### Ego Fairness
For evaluating the performance of the `llava-v1.6-mistral-7b` on ego fairness:
```
python main.py \
    --task  [driver_gender/driver_race/driver_age] \
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --image-folder [images-folder-of-nuscence] \
    --question-file ./qa-pairs/trustfulness/drivelm.json \
    --pattern-file ./qa-pairs/fairness/vehicle_qa.json \
    --answers-file [fairness-output-dir] \
    --subtype [female/male]/[caucasian/african_american/hispanic/native_american]/[young/middle_aged/elderly]
```

#### Scene Fairness
For evaluating the performance of the `llava-v1.6-mistral-7b` on scene fairness`:
```
python main.py \
    --task scene \
    --model-path  liuhaotian/llava-v1.6-mistral-7b \
    --image-folder [images-folder-of-nuscence] \
    --question-file ./qa-pairs/fairness/[pedestrian_qa/vehicle_qa].json \
    --answers-file [fairness-output-dir] 
```


## Contact
For help or issues using this project, please submit a GitHub issue.

For personal communication related to this project, please contact Shuo Xing (shuoxing@tamu.edu).

## Citation
We are more than happy if this code is helpful to your work. 
If you use our code or extend our work, please consider citing our paper:

```bibtex

@article{autotrust,
	author = {Xing, Shuo and Hua, Hongyuan and Gao, Xiangbo and Zhu, Shenzhe and Li, Renjie and Tian, Kexin and Li, Xiaopeng and Huang, Heng and Yang, Tianbao and Wang, Zhangyang and Zhou, Yang and Yao, Huaxiu and Tu, Zhengzhong},
	title = {{AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving}},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.15206},
	doi = {10.48550/arXiv.2412.15206}
}