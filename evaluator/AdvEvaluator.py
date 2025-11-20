from evaluator.BaseEvaluator import BaseEvaluator
import os
import re
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from evaluator.utils import list2dict
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union

# --- Imports for Attacks ---
from evaluator.whitebox_attacks import to_tanh_space, from_tanh_space, cw_loss

# --- Model Specific Imports ---
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.mm_utils_differentiable import process_images as process_images_differentiable
import EM_VLM4AD.eval_single as EM_VLM4AD

try:
    import dolphins.inference_image as dolphins
except:
    pass
from DriveLM import llama 
try:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
except ImportError:
    # import traceback; traceback.print_exc()
    pass
from Llama32.utils import preprocess_llama11B


class AdvEvaluator(BaseEvaluator):
    
    '''
    Generating while-box adversarial examples (PGD, BIM, CW) and evaluate the performance.
    '''
    
    def __init__(self, args, save_image=False):
        super().__init__(args)
        self.save_image = save_image
            
    def evaluate_white_box_attack(self, args=None):
        dataset_filtering_func = lambda x: x["question_type"] != "open-ended"
        self.answer_qa(args=args, dataset_filtering_func=dataset_filtering_func)
        
    def init_llama_adapter(self, args=None):
        super().init_llama_adapter(args)
        self.tokenizer = self.model.tokenizer
        # The original transform cannot be made differentiable, so we need to change the order of the transforms
        if self.transform:
            self.transform = T.Compose([
                self.transform.transforms[1],
                self.transform.transforms[0],
                self.transform.transforms[2]
            ])
            
    def init_blip2(self, args=None):
        super().init_blip2(args)
        self.tokenizer = self.processor.tokenizer
    
    def PGD_step(self, image_clean, image, data_grad, alpha=2/255, epsilon=16/255, max_v=1):
        """PGD / BIM step"""
        sign_data_grad = data_grad.sign()
        
        perturbed_image = image + alpha * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, max_v)
        
        clipped_perturb = torch.clamp(image_clean - perturbed_image, min=-epsilon, max=epsilon)
        perturbed_image = image_clean - clipped_perturb
        
        return perturbed_image   
        
    def init_llama_11B(self, args=None):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer     
        
    def get_loss(self, output, gt, criterion, task, args):
        # GT = {'multi-choice': ["A", "B", "C", "D"], 
        #       'yes-or-no': ["yes", 'Yes', "no", 'No']}
        
        if gt is not None:
            if hasattr(output, "logits"):
                logits_sft = output.logits[:, -1, :]
            elif type(output) == torch.Tensor:
                logits_sft = output[:, -1, :]
            elif type(output) == tuple:
                logits_sft = output[-1]
            else:
                raise ValueError("Invalid output")
            
            if task == "multi-choice":
                gts = ["A", "B", "C", "D"]
                if args.bce:
                    gt_label = torch.zeros(1, 4).cuda().float()
                    if gt not in gts: 
                        print(f"Warning: Invalid gt: {gt}. This is a bug need to be fixed")
                        gt = gts[0]
                    gt_label[0, gts.index(gt)] = 1
                else:
                    gt_label = torch.tensor(gts.index(gt)).cuda().long().unsqueeze(0).repeat(logits_sft.shape[0])
                
            elif task == "yes-or-no":
                if args.bce:
                    gts = ["yes", 'Yes', "no", 'No']
                    if gt not in gts: 
                        print(f"Warning: Invalid gt: {gt}. This is a bug need to be fixed")
                        gt = gts[0]
                    gt_label = torch.zeros(1, 4).cuda().float()
                    if gt in ['yes', 'Yes']:
                        gt_label[0, :2] = 1
                    elif gt in ["no", 'No']:
                        gt_label[0, 2:] = 1
                else:
                    gts1 = ["yes", "no"]
                    gts2 = ['Yes', 'No']
                    if gt in gts1:
                        gt_label = torch.tensor(gts1.index(gt)).cuda().long().unsqueeze(0).repeat(logits_sft.shape[0])
                        gts = gts1
                    elif gt in gts2:
                        gt_label = torch.tensor(gts2.index(gt)).cuda().long().unsqueeze(0).repeat(logits_sft.shape[0])
                        gts = gts2
            else:
                raise ValueError("Invalid task")
            
            token_ids = []
            for i in range(len(gts)):
                try:
                    token_id = self.tokenizer.encode(gts[i], add_special_tokens=False)
                except:
                    token_id = self.tokenizer.encode(gts[i], False, False)
                assert len(token_id) == 1
                token_ids.append(token_id[0])
            
            logits_mc = logits_sft[:, token_ids]
            pred = torch.argmax(logits_mc, dim=-1)
            loss = criterion(logits_mc, gt_label)
            
        else:
            raise ValueError("GT is required")
        
        return loss, (pred.item(), gt_label.argmax(dim=-1)[0].item() if args.bce else gt_label.item())
       
    def save_purterbed_image(self, image_perturbed, args):
        method = getattr(args, 'attack_method', 'PGD')
        perturbed_directory = os.path.join(
            args.image_folder,
            f"{self.image_file.split('/')[0]}_perturbed_{method}_{args.epsilon}_{args.alpha}_{args.num_iter}",
            *self.image_file.split('/')[1:-1]
        )
        os.makedirs(perturbed_directory, exist_ok=True)
        image_path_perturbed = os.path.join(perturbed_directory, self.image_file.split('/')[-1])
        cv2.imwrite(
            image_path_perturbed,
            cv2.cvtColor((image_perturbed * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        )
        
    # ------------------------------------------------------------------------------------
    # C&W Helper to get Target Index from GT String
    # ------------------------------------------------------------------------------------
    def get_target_idx_from_gt(self, gt, task):
        """Helper to map GT string to index 0-3 (MC) or 0-1 (Yes/No) for CW loss"""
        if task == "multi-choice":
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            return mapping.get(gt, 0) 
        elif task == "yes-or-no":
            if gt.lower() in ['yes']: return 0
            return 1
        return 0

    # ------------------------------------------------------------------------------------
    # DOLPHINS ATTACKS
    # ------------------------------------------------------------------------------------

    def CW_attack_dolphins(self, inputs, gt, task, iterations=100, lr=1e-2, args=None):
        # Prepare image
        size = self.image_processor.transforms[0].size
        mean = self.image_processor.transforms[-1].mean
        std = self.image_processor.transforms[-1].std
        image_path = os.path.join(args.image_folder, self.image_file)
        image = dolphins.get_image(image_path)
        image_tensor_ori = T.ToTensor()(image).to(self.device)[:3]
        
        # 1. Setup C&W variables
        w = to_tanh_space(image_tensor_ori).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=lr)
        image_tensor_clean = image_tensor_ori.clone().detach()
        
        # Target index for loss
        target_idx = torch.tensor([self.get_target_idx_from_gt(gt, task)]).to(self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            
            # 2. Reconstruct image
            adv_image = from_tanh_space(w)
            
            # 3. Preprocess for model
            image_tensor_resized = TF.resize(adv_image, size=(size,size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
            # Note: Ensure crop logic matches PGD if needed
            image_tensor_normalized = TF.normalize(image_tensor_resized, mean=mean, std=std)
            image_tensor = image_tensor_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0).half()
            
            # 4. Model Forward
            outputs = self.model(lang_x=inputs["input_ids"].to(self.device), vision_x=image_tensor, attention_mask=inputs["attention_mask"].to(self.device))
            
            # 5. Extract Logits for C&W
            logits_sft = outputs.logits[:, -1, :]
            if task == "multi-choice":
                 # [A, B, C, D]
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
            elif task == "yes-or-no":
                # [Yes, No]
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["yes", "no"]]
            else:
                token_ids = [] # Fallback

            relevant_logits = logits_sft[:, token_ids]
            
            # 6. Loss
            l2_loss = ((adv_image - image_tensor_clean) ** 2).sum()
            f_loss = cw_loss(relevant_logits, target_idx, kappa=0, target_is_gt=True)
            total_loss = l2_loss + f_loss # Can add constant 'c' if needed
            
            total_loss.backward()
            optimizer.step()
            
        # Final Return
        final_image = from_tanh_space(w).detach()
        # Process for return
        image_tensor_resized = TF.resize(final_image, size=(size,size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        image_tensor_normalized = TF.normalize(image_tensor_resized, mean=mean, std=std)
        image_tensor_processed = image_tensor_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0).half()
        
        image_numpy = final_image.cpu().detach().numpy().clip(0,1).transpose(1, 2, 0).astype('float32')
        return image_numpy, image_tensor_processed

    def PGD_attack_dolphins(self, inputs, gt, task, iterations=10, alpha=2/255, epsilon=16/255, args=None):
        size = self.image_processor.transforms[0].size
        mean = self.image_processor.transforms[-1].mean
        std = self.image_processor.transforms[-1].std
        image_path = os.path.join(args.image_folder, self.image_file)
        image = dolphins.get_image(image_path)
        image_tensor_ori = T.ToTensor()(image).to(self.device)[:3]
        image_tensor_ori.requires_grad_(True)
        image_tensor_ori_clean = image_tensor_ori.clone().detach().requires_grad_(False)
        
        # Initial crop/process
        image_tensor_resized = TF.resize(image_tensor_ori, size=(size,size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        # image_tensor_cropped = TF.center_crop(image_tensor_resized, output_size=(336, 336)) # Assuming standard logic
        image_tensor_normalized = TF.normalize(image_tensor_resized, mean=mean, std=std)
        image_tensor = image_tensor_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0).half()
        
        if args.bce: criterion = torch.nn.BCEWithLogitsLoss()
        else: criterion = torch.nn.CrossEntropyLoss()
            
        for _ in range(iterations):
            outputs = self.model(lang_x=inputs["input_ids"].to(self.device), vision_x=image_tensor, attention_mask=inputs["attention_mask"].to(self.device))
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]
            
            image_tensor_ori = self.PGD_step(image_tensor_ori_clean, image_tensor_ori, grad, alpha=alpha, epsilon=epsilon).detach()
            image_tensor_ori.requires_grad_(True)
            if image_tensor_ori.grad is not None: image_tensor_ori.grad.zero_()

            image_tensor_resized = TF.resize(image_tensor_ori, size=size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
            image_tensor_normalized = TF.normalize(image_tensor_resized, mean=mean, std=std)
            image_tensor = image_tensor_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0).half()
            torch.cuda.empty_cache()

        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')

        return image_perturbed, image_tensor

    # ------------------------------------------------------------------------------------
    # LLAVA ATTACKS
    # ------------------------------------------------------------------------------------

    def CW_attack_llava(self, input_ids, gt, task, iterations=100, lr=1e-2, args=None):
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).half().cuda().detach()
        
        w = to_tanh_space(image_tensor_ori).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=lr)
        image_tensor_clean = image_tensor_ori.clone().detach()
        target_idx = torch.tensor([self.get_target_idx_from_gt(gt, task)]).to(self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            adv_image = from_tanh_space(w)
            
            # Process
            image_tensor = process_images_differentiable([adv_image], self.image_processor, self.model.config)[0]
            
            # Forward
            outputs = self.model(input_ids, images=image_tensor, image_sizes=[image.size])
            
            # Logits
            logits_sft = outputs.logits[:, -1, :]
            if task == "multi-choice":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
            elif task == "yes-or-no":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["yes", "no"]]
            else: token_ids = []
            
            relevant_logits = logits_sft[:, token_ids]

            l2_loss = ((adv_image - image_tensor_clean) ** 2).sum()
            f_loss = cw_loss(relevant_logits, target_idx, kappa=0, target_is_gt=True)
            total_loss = l2_loss + f_loss
            
            total_loss.backward()
            optimizer.step()
            
        final_image = from_tanh_space(w).detach()
        image_tensor = process_images_differentiable([final_image], self.image_processor, self.model.config)[0]
        image_numpy = final_image.cpu().detach().numpy().clip(0,1).transpose(1, 2, 0).astype('float32')
        return image_numpy, image_tensor
             
    def PGD_attack_llava(self, input_ids, gt, task, iterations=10, alpha=2/255, epsilon=16/255, args=None):
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).half().cuda().detach()
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.detach().clone().requires_grad_(False)

        image_tensor = process_images_differentiable([image_tensor_ori], self.image_processor, self.model.config)[0]
        
        if args.bce: criterion = torch.nn.BCEWithLogitsLoss()
        else: criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            outputs = self.model(input_ids, images=image_tensor, image_sizes=[image.size])
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            image_tensor_ori = self.PGD_step(image_tensor_clean_ori, image_tensor_ori, grad, alpha=alpha, epsilon=epsilon).detach()
            image_tensor_ori.requires_grad_(True)
            if image_tensor_ori.grad is not None: image_tensor_ori.grad.zero_()

            image_tensor = process_images_differentiable([image_tensor_ori], self.image_processor, self.model.config)[0]
            torch.cuda.empty_cache()

        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')
        return image_perturbed, image_tensor
     
    # ------------------------------------------------------------------------------------
    # EM_VLM4AD ATTACKS
    # ------------------------------------------------------------------------------------

    def CW_attack_EM_VLM4AD(self, question, gt, task, iterations=100, lr=1e-2, args=None):
        # Helper transform for this model
        def transform(image_tensor, preprocessor=self.image_processor):
            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.unsqueeze(0)
            image_resized = TF.resize(image_tensor, size=preprocessor.transforms[0].size, interpolation=preprocessor.transforms[0].interpolation, antialias=preprocessor.transforms[0].antialias)
            image_resized = image_resized.squeeze(0)
            mean = torch.tensor(preprocessor.transforms[-1].mean).view(-1, 1, 1).to(self.device)
            std = torch.tensor(preprocessor.transforms[-1].std).view(-1, 1, 1).to(self.device)
            return (image_resized - mean) / std

        from torchvision.io import read_image, ImageReadMode
        question_encodings = self.tokenizer(question, padding=True, return_tensors="pt").input_ids.to(self.device)
        image_tensor_ori = read_image(os.path.join(args.image_folder, self.image_file), mode=ImageReadMode.RGB).float() / 255.0
        
        w = to_tanh_space(image_tensor_ori).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=lr)
        image_tensor_clean = image_tensor_ori.clone().detach()
        target_idx = torch.tensor([self.get_target_idx_from_gt(gt, task)]).to(self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            adv_image = from_tanh_space(w)
            image_tensor = transform(adv_image * 255).to(self.device) # Model expects scaled normalization
            
            outputs = self.model(question_encodings, image_tensor)
            logits_sft = outputs[-1] # Tuple output
            
            if task == "multi-choice":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
            elif task == "yes-or-no":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["yes", "no"]]
            else: token_ids = []
            relevant_logits = logits_sft[:, token_ids]

            l2_loss = ((adv_image - image_tensor_clean) ** 2).sum()
            f_loss = cw_loss(relevant_logits, target_idx, kappa=0, target_is_gt=True)
            total_loss = l2_loss + f_loss
            total_loss.backward()
            optimizer.step()
            
        final_image = from_tanh_space(w).detach()
        image_tensor = transform(final_image * 255).to(self.device)
        image_numpy = final_image.cpu().detach().numpy().clip(0,1).transpose(1, 2, 0).astype('float32')
        return image_numpy, image_tensor

    def PGD_attack_EM_VLM4AD(self, question, gt, task, iterations=10, alpha=2/255, epsilon=16/255, args=None):
        def transform(image_tensor, preprocessor=self.image_processor):
            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.unsqueeze(0) 
            image_resized = TF.resize(
                image_tensor, 
                size=preprocessor.transforms[0].size, 
                interpolation=preprocessor.transforms[0].interpolation, 
                antialias=preprocessor.transforms[0].antialias
            )
            image_resized = image_resized.squeeze(0) 
            mean = preprocessor.transforms[-1].mean
            mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
            std = preprocessor.transforms[-1].std
            std = torch.tensor(std).view(-1, 1, 1).to(self.device)
            image_normalized = (image_resized - mean) / std 
            return image_normalized

        from torchvision.io import read_image, ImageReadMode
        question_encodings = self.tokenizer(question, padding=True, return_tensors="pt").input_ids.to(self.device)
        image_tensor_ori = read_image(os.path.join(args.image_folder, self.image_file), mode=ImageReadMode.RGB).float()
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.clone().detach().requires_grad_(False)
        image_tensor = transform(image_tensor_ori).to(self.device)
        
        if args.bce: criterion = torch.nn.BCEWithLogitsLoss()
        else: criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            outputs = self.model(question_encodings, image_tensor)
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Note: Original code scaled by 255 here because read_image returns 0-255
            image_tensor_ori = self.PGD_step(
                image_tensor_clean_ori,
                image_tensor_ori,
                grad,
                alpha=alpha * 255,
                epsilon=epsilon * 255,
                max_v=255
            ).detach()
            
            image_tensor_ori.requires_grad_(True)
            if image_tensor_ori.grad is not None: image_tensor_ori.grad.zero_()
            image_tensor = transform(image_tensor_ori).to(self.device)
            torch.cuda.empty_cache()

        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed / 255.0, 0, 1) # Normalize for saving
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')
        return image_perturbed, image_tensor
    
    # ------------------------------------------------------------------------------------
    # LLAMA ADAPTER ATTACKS
    # ------------------------------------------------------------------------------------

    def CW_attack_llama_adapter(self, question, gt, task, iterations=100, lr=1e-2, args=None):
        def transform(image_tensor, preprocessor):
            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.unsqueeze(0)
            image_resized = TF.resize(image_tensor, size=preprocessor.transforms[1].size, interpolation=preprocessor.transforms[1].interpolation, max_size=preprocessor.transforms[1].max_size, antialias=preprocessor.transforms[1].antialias)
            image_resized = image_resized.squeeze(0)
            mean = torch.tensor(preprocessor.transforms[-1].mean).view(-1, 1, 1).to(self.device)
            std = torch.tensor(preprocessor.transforms[-1].std).view(-1, 1, 1).to(self.device)
            return (image_resized - mean) / std

        image = cv2.imread(os.path.join(args.image_folder, self.image_file))
        image = Image.fromarray(image)
        image_tensor_ori = TF.to_tensor(image).to(self.device)
        
        w = to_tanh_space(image_tensor_ori).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=lr)
        image_tensor_clean = image_tensor_ori.clone().detach()
        
        tokens = torch.tensor(self.model.tokenizer.encode(question[0], False, False)).unsqueeze(0).to(self.device)
        target_idx = torch.tensor([self.get_target_idx_from_gt(gt, task)]).to(self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            adv_image = from_tanh_space(w)
            
            if self.transform:
                image_tensor = transform(adv_image, self.transform).to(self.device).unsqueeze(0).unsqueeze(0)
            else: image_tensor = adv_image.unsqueeze(0).unsqueeze(0) # Fallback

            outputs = self.model.forward_outputs(tokens, image_tensor.to(self.device).half())
            logits_sft = outputs[:, -1, :]
            
            if task == "multi-choice":
                token_ids = [self.model.tokenizer.encode(c, False, False)[0] for c in ["A", "B", "C", "D"]]
            elif task == "yes-or-no":
                token_ids = [self.model.tokenizer.encode(c, False, False)[0] for c in ["yes", "no"]]
            else: token_ids = []
            relevant_logits = logits_sft[:, token_ids]

            l2_loss = ((adv_image - image_tensor_clean) ** 2).sum()
            f_loss = cw_loss(relevant_logits, target_idx, kappa=0, target_is_gt=True)
            total_loss = l2_loss + f_loss
            total_loss.backward()
            optimizer.step()
            
        final_image = from_tanh_space(w).detach()
        if self.transform:
             image_tensor = transform(final_image, self.transform).to(self.device).unsqueeze(0).unsqueeze(0)
        image_numpy = final_image.cpu().detach().numpy().clip(0,1).transpose(1, 2, 0).astype('float32')
        return image_numpy, image_tensor

    def PGD_attack_llama_adapter(self, question, gt, task, iterations=10, alpha=2/255, epsilon=16/255, args=None):
        def transform(image_tensor, preprocessor):
            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.unsqueeze(0)
            image_resized = TF.resize(image_tensor, 
                                      size=preprocessor.transforms[1].size, 
                                      interpolation=preprocessor.transforms[1].interpolation, 
                                      max_size=preprocessor.transforms[1].max_size,
                                      antialias=preprocessor.transforms[1].antialias)
            image_resized = image_resized.squeeze(0)
            mean = preprocessor.transforms[-1].mean
            mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
            std = preprocessor.transforms[-1].std
            std = torch.tensor(std).view(-1, 1, 1).to(self.device)
            image_normalized = (image_resized - mean) / std
            return image_normalized
        
        image = cv2.imread(os.path.join(args.image_folder, self.image_file))
        image = Image.fromarray(image)
        image_tensor_ori = TF.to_tensor(image).to(self.device)
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.clone().detach().requires_grad_(False)
        if self.transform:
            image_tensor = transform(image_tensor_ori, self.transform).to(self.device).unsqueeze(0).unsqueeze(0)
            
        if args.bce: criterion = torch.nn.BCEWithLogitsLoss()
        else: criterion = torch.nn.CrossEntropyLoss()
        
        tokens = torch.tensor(self.model.tokenizer.encode(question[0], False, False)).unsqueeze(0).to(self.device)
        
        for _ in range(iterations):
            outputs = self.model.forward_outputs(tokens, image_tensor.to(self.device).half())
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            image_tensor_ori = self.PGD_step(image_tensor_clean_ori, image_tensor_ori, grad, alpha=alpha, epsilon=epsilon).detach()
            image_tensor_ori.requires_grad_(True)
            if image_tensor_ori.grad is not None: image_tensor_ori.grad.zero_()
                
            if self.transform:
                image_tensor = transform(image_tensor_ori, self.transform).to(self.device).unsqueeze(0).unsqueeze(0)
            torch.cuda.empty_cache()

        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')
        return image_perturbed, image_tensor
             
    # ------------------------------------------------------------------------------------
    # DRIVELM / BLIP2 ATTACKS
    # ------------------------------------------------------------------------------------

    def CW_attack_drivelm_agent(self, question, gt, task, iterations=100, lr=1e-2, args=None):
        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).to(self.device)
        
        w = to_tanh_space(image_tensor_ori).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=lr)
        image_tensor_clean = image_tensor_ori.clone().detach()
        
        size = (self.processor.image_processor.size['height'], self.processor.image_processor.size['width'])
        mean = torch.tensor(self.processor.image_processor.image_mean).view(-1, 1, 1).to(self.device)
        std = torch.tensor(self.processor.image_processor.image_std).view(-1, 1, 1).to(self.device)
        
        target_idx = torch.tensor([self.get_target_idx_from_gt(gt, task)]).to(self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            adv_image = from_tanh_space(w)
            
            image_tensor = adv_image.unsqueeze(0)
            image_tensor = TF.resize(image_tensor, size=size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            image_tensor = (image_tensor - mean) / std
            
            decoder_input_ids = torch.ones((image_tensor.shape[0], 1), dtype=torch.long, device=self.device)
            outputs = self.model.base_model(pixel_values=image_tensor.to(self.device), input_ids=question, decoder_input_ids=decoder_input_ids)
            logits_sft = outputs.logits[:, -1, :]
            
            if task == "multi-choice":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
            elif task == "yes-or-no":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["yes", "no"]]
            else: token_ids = []
            relevant_logits = logits_sft[:, token_ids]

            l2_loss = ((adv_image - image_tensor_clean) ** 2).sum()
            f_loss = cw_loss(relevant_logits, target_idx, kappa=0, target_is_gt=True)
            total_loss = l2_loss + f_loss
            total_loss.backward()
            optimizer.step()
            
        final_image = from_tanh_space(w).detach()
        image_tensor = final_image.unsqueeze(0)
        image_tensor = TF.resize(image_tensor, size=size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        image_tensor = (image_tensor - mean) / std
        image_numpy = final_image.cpu().detach().numpy().clip(0,1).transpose(1, 2, 0).astype('float32')
        return image_numpy, image_tensor

    def PGD_attack_drivelm_agent(self, question, gt, task, iterations=10, alpha=2/255, epsilon=16/255, args=None):
        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).to(self.device)
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.clone().detach().requires_grad_(False)
        
        size = (self.processor.image_processor.size['height'], self.processor.image_processor.size['width'])
        mean = torch.tensor(self.processor.image_processor.image_mean).view(-1, 1, 1).to(self.device)
        std = torch.tensor(self.processor.image_processor.image_std).view(-1, 1, 1).to(self.device)
        
        image_tensor = image_tensor_ori.unsqueeze(0) 
        image_tensor = TF.resize(image_tensor, size=size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        image_tensor = (image_tensor - mean) / std  
        
        if args.bce: criterion = torch.nn.BCEWithLogitsLoss()
        else: criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            decoder_input_ids = torch.ones((image_tensor.shape[0], 1), dtype=torch.long, device=self.device)
            outputs = self.model.base_model(pixel_values=image_tensor.to(self.device), 
                                            input_ids=question, 
                                            decoder_input_ids=decoder_input_ids)
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            image_tensor_ori = self.PGD_step(image_tensor_clean_ori, image_tensor_ori, grad, alpha=alpha, epsilon=epsilon).detach()
            image_tensor_ori.requires_grad_(True)
            if image_tensor_ori.grad is not None: image_tensor_ori.grad.zero_()
                
            image_tensor = image_tensor_ori.unsqueeze(0)
            image_tensor = TF.resize(image_tensor, size=size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            image_tensor = (image_tensor - mean) / std  
            torch.cuda.empty_cache()

        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')
        return image_perturbed, image_tensor   
    
    # ------------------------------------------------------------------------------------
    # LLAMA 11B ATTACKS
    # ------------------------------------------------------------------------------------

    def CW_attack_llama_11B(self, inputs, gt, task, iterations=100, lr=1e-2, args=None):
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).half().cuda().detach()
        
        w = to_tanh_space(image_tensor_ori).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=lr)
        image_tensor_clean = image_tensor_ori.clone().detach()
        target_idx = torch.tensor([self.get_target_idx_from_gt(gt, task)]).to(self.device)

        for _ in range(iterations):
            optimizer.zero_grad()
            adv_image = from_tanh_space(w)
            image_tensor = preprocess_llama11B(adv_image, self.processor.image_processor).unsqueeze(0).unsqueeze(0)
            
            inputs['pixel_values'] = image_tensor
            outputs = self.model(**inputs)
            logits_sft = outputs.logits[:, -1, :]
            
            if task == "multi-choice":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
            elif task == "yes-or-no":
                token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["yes", "no"]]
            else: token_ids = []
            relevant_logits = logits_sft[:, token_ids]

            l2_loss = ((adv_image - image_tensor_clean) ** 2).sum()
            f_loss = cw_loss(relevant_logits, target_idx, kappa=0, target_is_gt=True)
            total_loss = l2_loss + f_loss
            total_loss.backward()
            optimizer.step()

        final_image = from_tanh_space(w).detach()
        image_tensor = preprocess_llama11B(final_image, self.processor.image_processor).unsqueeze(0).unsqueeze(0)
        image_numpy = final_image.cpu().detach().numpy().clip(0,1).transpose(1, 2, 0).astype('float32')
        return image_numpy, image_tensor

    def PGD_attack_llama_11B(self, inputs, gt, task, iterations=10, alpha=2/255, epsilon=16/255, args=None):
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        
        image_tensor_ori = T.ToTensor()(image).half().cuda().detach()
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.detach().clone().requires_grad_(False)
        image_tensor = preprocess_llama11B(image_tensor_ori, self.processor.image_processor).unsqueeze(0).unsqueeze(0)
            
        if args.bce: criterion = torch.nn.BCEWithLogitsLoss()
        else: criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            inputs['pixel_values'] = image_tensor
            outputs = self.model(**inputs)
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)

            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]
            image_tensor_ori = self.PGD_step(image_tensor_clean_ori, image_tensor_ori, grad, alpha=alpha, epsilon=epsilon).detach()
            image_tensor_ori.requires_grad_(True)
            if image_tensor_ori.grad is not None: image_tensor_ori.grad.zero_()

            image_tensor = preprocess_llama11B(image_tensor_ori, self.processor.image_processor).unsqueeze(0).unsqueeze(0)
            torch.cuda.empty_cache()

        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')
        return image_perturbed, image_tensor
    
    # ------------------------------------------------------------------------------------
    # EVAL METHODS (Routing logic added)
    # ------------------------------------------------------------------------------------

    def eval_llama_11B_model(self, item, add_prompt=None, args=None, gt=None):
        self.image_file = item["image_path"]
        self.qs = item["question"]
        if not add_prompt is None: self.qs += add_prompt
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.qs}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(self.device)  

        for param in self.model.parameters(): param.requires_grad_(False)
        self.model.eval()
        
        method = getattr(args, 'attack_method', 'PGD')
        if method == 'CW':
            image_perturbed, image_perturbed_processed = self.CW_attack_llama_11B(inputs, gt, item["question_type"], iterations=100, args=args)
        elif method == 'BIM':
            image_perturbed, image_perturbed_processed = self.PGD_attack_llama_11B(inputs, gt, item["question_type"], iterations=10, alpha=(8/255)/10, epsilon=8/255, args=args)
        else: # PGD
            image_perturbed, image_perturbed_processed = self.PGD_attack_llama_11B(inputs, gt, item["question_type"], iterations=args.num_iter, alpha=args.alpha/255, epsilon=args.epsilon/255, args=args)
        
        inputs["pixel_values"] = image_perturbed_processed
            
        if self.save_image: self.save_purterbed_image(image_perturbed, args)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs,
                                    do_sample=True if args.temperature > 0 else False,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    num_beams=args.num_beams,
                                    max_new_tokens=1024,
                                    use_cache=True)
        
        outputs = self.processor.decode(outputs[0]).replace(input_text, "").replace("<|eot_id|>","").strip()
        return outputs
           
    def eval_llava_model(self, item, add_prompt=None, args=None, gt=None):
        self.image_file = item["image_path"]
        self.qs = item["question"]
        if not add_prompt is None: self.qs += add_prompt

        if IMAGE_PLACEHOLDER in self.qs:
            if self.model.config.mm_use_im_start_end: self.qs = re.sub(IMAGE_PLACEHOLDER, self.image_token_se, self.qs)
            else: self.qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, self.qs)
        else:
            if self.model.config.mm_use_im_start_end: self.qs = self.image_token_se + "\n" + self.qs
            else: self.qs = DEFAULT_IMAGE_TOKEN + "\n" + self.qs
                
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids.requires_grad_(False)     

        for param in self.model.parameters(): param.requires_grad_(False)
        self.model.eval()

        method = getattr(args, 'attack_method', 'PGD')
        if method == 'CW':
            image_perturbed, image_perturbed_processed = self.CW_attack_llava(input_ids, gt, item["question_type"], iterations=100, args=args)
        elif method == 'BIM':
            image_perturbed, image_perturbed_processed = self.PGD_attack_llava(input_ids, gt, item["question_type"], iterations=10, alpha=(8/255)/10, epsilon=8/255, args=args)
        else:
            image_perturbed, image_perturbed_processed = self.PGD_attack_llava(input_ids, gt, item["question_type"], iterations=args.num_iter, alpha=args.alpha/255, epsilon=args.epsilon/255, args=args)
            
        if self.save_image: self.save_purterbed_image(image_perturbed, args)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_perturbed_processed,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def eval_dolphins_model(self, item, add_prompt=None,args=None, gt=None):
        self.image_file = item["image_path"]
        question = item["question"]
        self.qs = question
        if add_prompt: question += add_prompt

        vision_x, inputs = dolphins.get_model_inputs(os.path.join(args.image_folder, self.image_file), question, self.model, self.image_processor, self.tokenizer)
        for param in self.model.parameters(): param.requires_grad_(False)
        self.model.eval()
        
        method = getattr(args, 'attack_method', 'PGD')
        if method == 'CW':
            image_perturbed, image_perturbed_processed = self.CW_attack_dolphins(inputs, gt, item["question_type"], iterations=100, args=args)
        elif method == 'BIM':
            image_perturbed, image_perturbed_processed = self.PGD_attack_dolphins(inputs, gt, item["question_type"], iterations=10, alpha=(8/255)/10, epsilon=8/255, args=args)
        else:
            image_perturbed, image_perturbed_processed = self.PGD_attack_dolphins(inputs, gt, item["question_type"], iterations=args.num_iter, alpha=args.alpha/255, epsilon=args.epsilon/255, args=args)
        
        if self.save_image: self.save_purterbed_image(image_perturbed, args)
        
        generation_kwargs = {'max_new_tokens': 512, 'temperature': 1, 'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1, 'do_sample': False, 'early_stopping': True}
        output_ids = self.model.generate(vision_x=image_perturbed_processed, lang_x=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].cuda(), num_beams=3, **generation_kwargs)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.split("GPT:")[1].strip()
        return outputs
    
    def eval_EM_VLM4AD_model(self, item, add_prompt=None, ignore_type=False, args=None, gt=None):
        self.image_file = item["image_path"]
        question = item["question"]
        if add_prompt:
            question += add_prompt
            self.qs = question
            
        for param in self.model.parameters(): param.requires_grad_(False)
        self.model.eval()
        
        method = getattr(args, 'attack_method', 'PGD')
        if method == 'CW':
            image_perturbed, image_perturbed_processed = self.CW_attack_EM_VLM4AD(question, gt, item["question_type"], iterations=100, args=args)
        elif method == 'BIM':
            image_perturbed, image_perturbed_processed = self.PGD_attack_EM_VLM4AD(question, gt, item["question_type"], iterations=10, alpha=(8/255)/10, epsilon=8/255, args=args)
        else:
            image_perturbed, image_perturbed_processed = self.PGD_attack_EM_VLM4AD(question, gt, item["question_type"], iterations=args.num_iter, alpha=args.alpha/255, epsilon=args.epsilon/255, args=args)
            
        if self.save_image: self.save_purterbed_image(image_perturbed, args)

        question_encodings = self.tokenizer(question, padding=True, return_tensors="pt").input_ids.to(self.device)

        if ignore_type:
            text_output = EM_VLM4AD.val_model_without_preprocess(self.model, self.tokenizer, question, image_perturbed_processed)[0]
        else:
            if item['question_type'] == 'open-ended':
                text_output = EM_VLM4AD.val_model_without_preprocess(self.model, self.tokenizer, question, image_perturbed_processed)[0]
            else:
                embed_output = self.model(question_encodings, image_perturbed_processed)
                if item['question_type'] == 'multi-choice':
                    label_pool = ['A', 'B', 'C', 'D']
                    token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in label_pool]
                    logits_sft = F.softmax(embed_output.logits[:,-1,:], dim=-1)
                    logits_A_B_C_D = logits_sft[:, token_ids].cpu()
                    pred_index = logits_A_B_C_D.argmax(dim=-1).item()
                    text_output = label_pool[pred_index]
                elif item['question_type'] == 'yes-or-no':
                    label_pool = ['Yes', 'No']
                    token_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ['yes', 'no']]
                    logits_sft = F.softmax(embed_output.logits[:,-1,:], dim=-1)
                    logits_A_B = logits_sft[:, token_ids].cpu()
                    pred_index = logits_A_B.argmax(dim=-1).item()
                    text_output = label_pool[pred_index]
        return text_output
    
    def eval_blip2(self, item, add_prompt=None, args=None, gt=None):
        self.image_file = item["image_path"]
        self.qs = item["question"]
        if not add_prompt is None: self.qs += add_prompt
            
        for param in self.model.parameters(): param.requires_grad_(False)
        self.model.eval()
        
        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        inputs = self.processor(image, self.qs, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        method = getattr(args, 'attack_method', 'PGD')
        if method == 'CW':
            image_perturbed, image_perturbed_processed = self.CW_attack_drivelm_agent(input_ids, gt, item["question_type"], iterations=100, args=args)
        elif method == 'BIM':
            image_perturbed, image_perturbed_processed = self.PGD_attack_drivelm_agent(input_ids, gt, item["question_type"], iterations=10, alpha=(8/255)/10, epsilon=8/255, args=args)
        else:
            image_perturbed, image_perturbed_processed = self.PGD_attack_drivelm_agent(input_ids, gt, item["question_type"], iterations=args.num_iter, alpha=args.alpha/255, epsilon=args.epsilon/255, args=args)
            
        if self.save_image: self.save_purterbed_image(image_perturbed, args)    

        with torch.inference_mode():
            output_ids = self.model.generate(
                pixel_values=image_perturbed_processed,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=args.temperature,
                )
        outputs = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return outputs
    
    def eval_llama_adapter(self, item, add_prompt=None, args=None, gt=None):
        self.image_file = item["image_path"]
        self.qs = item["question"]
        if not add_prompt is None: self.qs += add_prompt
        
        for param in self.model.parameters(): param.requires_grad_(False)
        self.model.eval()
        
        prompt = [llama.format_prompt(self.qs)]
        
        method = getattr(args, 'attack_method', 'PGD')
        if method == 'CW':
            image_perturbed, image_perturbed_processed = self.CW_attack_llama_adapter(prompt, gt, item["question_type"], iterations=100, args=args)
        elif method == 'BIM':
            image_perturbed, image_perturbed_processed = self.PGD_attack_llama_adapter(prompt, gt, item["question_type"], iterations=10, alpha=(8/255)/10, epsilon=8/255, args=args)
        else:
            image_perturbed, image_perturbed_processed = self.PGD_attack_llama_adapter(prompt, gt, item["question_type"], iterations=args.num_iter, alpha=args.alpha/255, epsilon=args.epsilon/255, args=args)
        
        outputs = self.model.generate(image_perturbed_processed.to(self.device), prompt, temperature=args.temperature)
        return outputs
         
    def answer_qa(self, prefix=None, suffix=None, question=None, ignore_type=False, dataset_filtering_func=lambda x: True, args=None):
        self.answer_dict_list = []
        with open(args.gt_file, "r") as file:
            gts = json.load(file)
            gts = list2dict(gts)
        
        if "llava" in self.model_name:
            self.init_llava(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item): continue
                if prefix is not None:
                    if isinstance(prefix, dict): prefix_i = prefix[str(item["question_id"])]
                    elif isinstance(prefix, str): prefix_i = prefix
                    else: raise ValueError("Invalid prefix type")
                else: prefix_i = None
                
                item, add_prompt = self.process_input(item, prefix=prefix_i, suffix=suffix, question=question)
                response = self.eval_llava_model(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])

                item["question"] = self.qs
                ans_dict = {"llava_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
                
        elif 'Dolphins' in self.model_name:
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item): continue
                if prefix is not None:
                    if isinstance(prefix, dict): prefix_i = prefix[str(item["question_id"])]
                    elif isinstance(prefix, str): prefix_i = prefix
                    else: raise ValueError("Invalid prefix type")
                else: prefix_i = None
                gt = gts[item["question_id"]]['answer']
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question)
                response = self.eval_dolphins_model(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"dolphins_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
                
        elif 'EM_VLM4AD' in self.model_name:
            self.init_EM_VLM4AD()
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item): continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_EM_VLM4AD_model(item, add_prompt=add_prompt, ignore_type=ignore_type, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"EM_VLM4AD_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)    
                
        elif "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item): continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_blip2(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"drivelm_agent_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item) 
                
        elif "llama_adapter" in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item): continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llama_adapter(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"llama_adapter_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item) 
        
        elif "Llama-3.2-11B-Vision" in self.model_name:
            self.init_llama_11B(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item): continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llama_11B_model(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"llama_adapter_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)    
            
        else:
            raise ValueError("Invalid model name")
        
        os.makedirs(os.path.dirname(self.answers_file), exist_ok=True)
        with open(self.answers_file, 'w') as json_file:
            json.dump(self.answer_dict_list, json_file, indent=4)