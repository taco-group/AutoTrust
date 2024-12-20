from evaluator.BaseEvaluator import BaseEvaluator
import os
import re
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluator.utils import list2dict
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union

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
    import traceback; traceback.print_exc()
from Llama32.utils import preprocess_llama11B


class AdvEvaluator(BaseEvaluator):
    
    '''
    Generating while-box adversarial examples for the models and evaluate the performance of the models on the white box adversarial attack.
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
        """PGD step

        Args:
            image_clean (torch.Tensor): Clean Image
            image (torch.Tensor): Current Image
            data_grad (torch.Tensor): Gradient of the image
            epsilon (float, optional): Step size. Defaults to 2/255.
            max_p (float, optional): Max perturbation. Defaults to 16/255.

        Returns:
            torch.Tensor: Perturbed Image
        """
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
                # For some models, such as llama_adapter, the output may not have the logits attribute
                logits_sft = output[:, -1, :]
            elif type(output) == tuple:
                # EM_VLM4AD
                logits_sft = output[-1]
            else:
                raise ValueError("Invalid output")
            
            if task == "multi-choice":
                gts = ["A", "B", "C", "D"]
                # assert gt in gts, f"Invalid gt: {gt}"
                if args.bce:
                    gt_label = torch.zeros(1, 4).cuda().float()
                    if gt not in gts: 
                        print(f"Warning: Invalid gt: {gt}. This is a bug need to be fixed")
                        gt = gts[0]
                    gt_label[0, gts.index(gt)] = 1
                else:
                    gt_label = torch.tensor(gts.index(gt)).cuda().long().unsqueeze(0).repeat(logits_sft.shape[0])
                
            elif task == "yes-or-no":
                
                # assert gt in gts, f"Invalid gt: {gt}"
                if args.bce:
                    gts = ["yes", 'Yes', "no", 'No']
                    if gt not in gts: 
                        print(f"Warning: Invalid gt: {gt}. This is a bug need to be fixed")
                        gt = gts[0]
                    # for yes or no questions, both capital and lower case are considered the same
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
            
            if args.bce:
                isinstance(criterion, torch.nn.BCEWithLogitsLoss)
            else:
                isinstance(criterion, torch.nn.CrossEntropyLoss)
            
            token_ids = []
            for i in range(len(gts)):
                try:
                    token_id = self.tokenizer.encode(gts[i], add_special_tokens=False)
                except:
                    # For some models, such as llama_adapter, the tokenizer may not have the add_special_tokens argument
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
        perturbed_directory = os.path.join(
            args.image_folder,
            f"{self.image_file.split('/')[0]}_perturbed_{args.epsilon}_{args.alpha}_{args.num_iter}_llama_11B",
            *self.image_file.split('/')[1:-1]
        )
        os.makedirs(perturbed_directory, exist_ok=True)
        image_path_perturbed = os.path.join(perturbed_directory, self.image_file.split('/')[-1])
        cv2.imwrite(
            image_path_perturbed,
            cv2.cvtColor((image_perturbed * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        )
        
    def PGD_attack_dolphins(self,
                        inputs,
                        gt,
                        task,
                        iterations=10,
                        alpha=2/255,
                        epsilon=16/255,
                        args=None):
        
        
        size = self.image_processor.transforms[0].size
        mean = self.image_processor.transforms[-1].mean
        std = self.image_processor.transforms[-1].std
        image_path = os.path.join(args.image_folder, self.image_file)
        image = dolphins.get_image(image_path)
        image_tensor_ori = T.ToTensor()(image).to(self.device)[:3]
        image_tensor_ori.requires_grad_(True)
        image_tensor_ori_clean = image_tensor_ori.clone().detach().requires_grad_(False)
        image_tensor_resized = TF.resize(image_tensor_ori, size=(size,size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        # image_tensor_cropped = TF.center_crop(image_tensor_resized, output_size=(336, 336))
        image_tensor_normalized = TF.normalize(image_tensor_cropped, mean=mean, std=std)
        image_tensor = image_tensor_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0).half()
        
        # image_reference, _ = dolphins.get_model_inputs(os.path.join(args.image_folder, image_path), "question place holder", self.model, self.image_processor, self.tokenizer)
        # TODO: Do not pass
        # assert torch.allclose(image_tensor, image_reference, atol=1e-5), "Image tensor is not equal to the reference image tensor"
        
         # Define loss criterion
        if args.bce:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
            
        for _ in range(iterations):
            # Compute outputs with gradients for the perturbed image
            outputs = self.model(lang_x=inputs["input_ids"].to(self.device), vision_x=image_tensor, attention_mask=inputs["attention_mask"].to(self.device))
            
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)

            # Compute gradients with respect to image_tensor_ori
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Update the perturbed image using FGSM attack
            image_tensor_ori = self.PGD_step(
                image_tensor_ori_clean,
                image_tensor_ori,
                grad,
                alpha=alpha,
                epsilon=epsilon
            ).detach()
            
            image_tensor_ori.requires_grad_(True)

            # Zero gradients of the perturbed image
            if image_tensor_ori.grad is not None:
                image_tensor_ori.grad.zero_()

            # Re-process the perturbed image
            image_tensor_resized = TF.resize(image_tensor_ori, size=size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
            image_tensor_cropped = TF.center_crop(image_tensor_resized, output_size=(336, 336))
            image_tensor_normalized = TF.normalize(image_tensor_cropped, mean=mean, std=std)
            image_tensor = image_tensor_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0).half()
            
            torch.cuda.empty_cache()

        # Convert the perturbed tensor to an image and save it
        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')

        return image_perturbed, image_tensor
             
    def PGD_attack_llava(self, 
                   input_ids, 
                   gt, 
                   task,
                   iterations=10, 
                   alpha=2/255, 
                   epsilon=16/255,
                   args=None):
        
        # Load and preprocess the image
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).half().cuda().detach()
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.detach().clone().requires_grad_(False)

        # Process images
        image_tensor = process_images_differentiable([image_tensor_ori], self.image_processor, self.model.config)[0]
        # Define loss criterion
        if args.bce:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            # Compute outputs with gradients for the perturbed image
            outputs = self.model(input_ids, images=image_tensor, image_sizes=[image.size])
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            
            
            # Compute gradients with respect to image_tensor_ori
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Update the perturbed image using FGSM attack
            image_tensor_ori = self.PGD_step(
                image_tensor_clean_ori,
                image_tensor_ori,
                grad,
                alpha=alpha,
                epsilon=epsilon
            ).detach()
            
            image_tensor_ori.requires_grad_(True)

            # Zero gradients of the perturbed image
            if image_tensor_ori.grad is not None:
                image_tensor_ori.grad.zero_()

            # Re-process the perturbed image
            image_tensor = process_images_differentiable(
                [image_tensor_ori],
                self.image_processor,
                self.model.config
            )[0]
            
            torch.cuda.empty_cache()

        # Convert the perturbed tensor to an image and save it
        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')

        return image_perturbed, image_tensor
     
    def PGD_attack_EM_VLM4AD(self,
                        question,
                        gt,
                        task,
                        iterations=10,
                        alpha=2/255,
                        epsilon=16/255,
                        args=None):
        
        # a function to transform the image tensor with computation graph
        def transform(image_tensor, preprocessor=self.image_processor):
            image_tensor = image_tensor.to(self.device)

            # Resize the image to (224, 224)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
            # F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            image_resized = TF.resize(
                image_tensor, 
                size=preprocessor.transforms[0].size, 
                interpolation=preprocessor.transforms[0].interpolation, 
                antialias=preprocessor.transforms[0].antialias
            )
            image_resized = image_resized.squeeze(0)  # Remove batch dimension -> [C, 224, 224]

            # Normalize the image
            mean = preprocessor.transforms[-1].mean
            mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
            std = preprocessor.transforms[-1].std
            std = torch.tensor(std).view(-1, 1, 1).to(self.device)
            image_normalized = (image_resized - mean) / std  # Values in range [-1, 1]

            return image_normalized  # Tensor of shape [C, 224, 224]
    
            
        from torchvision.io import read_image, ImageReadMode
        question_encodings = self.tokenizer(question, padding=True, return_tensors="pt").input_ids.to(self.device)
        image_tensor_ori = read_image(os.path.join(args.image_folder, self.image_file), mode=ImageReadMode.RGB).float()
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.clone().detach().requires_grad_(False)
        image_tensor = transform(image_tensor_ori).to(self.device)
        
        image_reference = EM_VLM4AD.preprocess_image(os.path.join(args.image_folder, self.image_file), self.image_processor)
        if not torch.allclose(image_tensor, image_reference, atol=1e-5):
            print(f"Warning: Image tensor is not equal to the reference image tensor, {torch.abs(image_tensor - image_reference).max()}")
        
        # Define loss criterion
        if args.bce:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            # Compute outputs with gradients for the perturbed image
            outputs = self.model(question_encodings, image_tensor)
            
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)

            # Compute gradients with respect to image_tensor_ori
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Update the perturbed image using FGSM attack
            image_tensor_ori = self.PGD_step(
                image_tensor_clean_ori,
                image_tensor_ori,
                grad,
                alpha=alpha * 255,
                epsilon=epsilon * 255,
                max_v=255
            ).detach()
            
            image_tensor_ori.requires_grad_(True)

            # Zero gradients of the perturbed image
            if image_tensor_ori.grad is not None:
                image_tensor_ori.grad.zero_()
                
            image_tensor = transform(image_tensor_ori).to(self.device)
            
            torch.cuda.empty_cache()

        # Convert the perturbed tensor to an image and save it
        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')

        return image_perturbed, image_tensor
    
    def PGD_attack_llama_adapter(self,
                        question,
                        gt,
                        task,
                        iterations=10,
                        alpha=2/255,
                        epsilon=16/255,
                        args=None):
        
        # a function to transform the image tensor with computation graph
        def transform(image_tensor, preprocessor):
            image_tensor = image_tensor.to(self.device)

            # Resize the image to (224, 224)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
            # F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            image_resized = TF.resize(image_tensor, 
                                      size=preprocessor.transforms[1].size, 
                                      interpolation=preprocessor.transforms[1].interpolation, 
                                      max_size=preprocessor.transforms[1].max_size,
                                      antialias=preprocessor.transforms[1].antialias)
            image_resized = image_resized.squeeze(0)  # Remove batch dimension -> [C, 224, 224]

            # Normalize the image
            mean = preprocessor.transforms[-1].mean
            mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
            std = preprocessor.transforms[-1].std
            std = torch.tensor(std).view(-1, 1, 1).to(self.device)
            image_normalized = (image_resized - mean) / std  # Values in range [-1, 1]

            return image_normalized  # Tensor of shape [C, 224, 224]
        
        image = cv2.imread(os.path.join(args.image_folder, self.image_file))
        image = Image.fromarray(image)
        image_tensor_ori = TF.to_tensor(image).to(self.device)
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.clone().detach().requires_grad_(False)
        if self.transform:
            image_tensor = transform(image_tensor_ori, self.transform).to(self.device).unsqueeze(0).unsqueeze(0)
            
            
        # To check if the transform function is correct
        if self.transform:
            image_reference = self.transform(image)
        image_reference = image_reference.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Passed
        # assert torch.allclose(image_tensor, image_reference, atol=1e-5), "Image tensor is not equal to the reference image tensor"
            
            
            
        # Define loss criterion
        if args.bce:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        tokens = torch.tensor(self.model.tokenizer.encode(question[0], False, False)).unsqueeze(0).to(self.device)
        
        for _ in range(iterations):
            # Compute outputs with gradients for the perturbed image
            outputs = self.model.forward_outputs(tokens, image_tensor.to(self.device).half())
            # outputs = self.model(question_encodings, image_tensor)
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)

            # Compute gradients with respect to image_tensor_ori
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Update the perturbed image using FGSM attack
            image_tensor_ori = self.PGD_step(
                image_tensor_clean_ori,
                image_tensor_ori,
                grad,
                alpha=alpha,
                epsilon=epsilon
            ).detach()
            
            image_tensor_ori.requires_grad_(True)

            # Zero gradients of the perturbed image
            if image_tensor_ori.grad is not None:
                image_tensor_ori.grad.zero_()
                
            if self.transform:
                image_tensor = transform(image_tensor_ori, self.transform).to(self.device).unsqueeze(0).unsqueeze(0)
            
            torch.cuda.empty_cache()

        # Convert the perturbed tensor to an image and save it
        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')
        
        return image_perturbed, image_tensor
             
    def PGD_attack_drivelm_agent(self,
                        question,
                        gt,
                        task,
                        iterations=10,
                        alpha=2/255,
                        epsilon=16/255,
                        args=None):

        
        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        image_tensor_ori = T.ToTensor()(image).to(self.device)
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.clone().detach().requires_grad_(False)
        
        size = (self.processor.image_processor.size['height'], self.processor.image_processor.size['width'])
        mean = torch.tensor(self.processor.image_processor.image_mean).view(-1, 1, 1).to(self.device)
        std = torch.tensor(self.processor.image_processor.image_std).view(-1, 1, 1).to(self.device)
        
        image_tensor = image_tensor_ori.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        image_tensor = TF.resize(image_tensor, size=size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        image_tensor = (image_tensor - mean) / std  # Values in range [-1, 1]  
        
        
        image_reference = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        image_reference = self.processor(image_reference, self.qs, return_tensors="pt")['pixel_values'].to(self.device)
        
        # TODO: Not Passed
        # assert torch.allclose(image_tensor, image_reference, atol=1e-5), "Image tensor is not equal to the reference image tensor"
        
        # Define loss criterion
        if args.bce:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        
        
        for _ in range(iterations):
            # Compute outputs with gradients for the perturbed image
            decoder_input_ids = torch.ones((image_tensor.shape[0], 1), dtype=torch.long, device=self.device)
            outputs = self.model.base_model(pixel_values=image_tensor.to(self.device), 
                                            input_ids=question, 
                                            decoder_input_ids=decoder_input_ids)
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)
            
            
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Update the perturbed image using FGSM attack
            image_tensor_ori = self.PGD_step(
                image_tensor_clean_ori,
                image_tensor_ori,
                grad,
                alpha=alpha,
                epsilon=epsilon
            ).detach()
            
            image_tensor_ori.requires_grad_(True)

            # Zero gradients of the perturbed image
            if image_tensor_ori.grad is not None:
                image_tensor_ori.grad.zero_()
                
            image_tensor = image_tensor_ori.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
            image_tensor = TF.resize(image_tensor, size=size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            image_tensor = (image_tensor - mean) / std  # Values in range [-1, 1]  
            
            torch.cuda.empty_cache()

        # Convert the perturbed tensor to an image and save it
        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')

        return image_perturbed, image_tensor   
         
    def PGD_attack_llama_11B(self, 
                   inputs,
                   gt, 
                   task,
                   iterations=10, 
                   alpha=2/255, 
                   epsilon=16/255,
                   args=None):
        
        # Load and preprocess the image
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')
        
        image_tensor_ori = T.ToTensor()(image).half().cuda().detach()
        image_tensor_ori.requires_grad_(True)
        image_tensor_clean_ori = image_tensor_ori.detach().clone().requires_grad_(False)
        image_tensor = preprocess_llama11B(image_tensor_ori, self.processor.image_processor).unsqueeze(0).unsqueeze(0)
            
            

        
        # Define loss criterion
        if args.bce:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(iterations):
            inputs['pixel_values'] = image_tensor
            # Compute outputs with gradients for the perturbed image
            outputs = self.model(**inputs)
            loss, (pred_idx, gt_idx) = self.get_loss(outputs, gt, criterion, task, args)

            # Compute gradients with respect to image_tensor_ori
            grad = torch.autograd.grad(loss, image_tensor_ori, retain_graph=False, create_graph=False)[0]

            # Update the perturbed image using FGSM attack
            image_tensor_ori = self.PGD_step(
                image_tensor_clean_ori,
                image_tensor_ori,
                grad,
                alpha=alpha,
                epsilon=epsilon
            ).detach()
            
            image_tensor_ori.requires_grad_(True)

            # Zero gradients of the perturbed image
            if image_tensor_ori.grad is not None:
                image_tensor_ori.grad.zero_()

            # Re-process the perturbed image
            image_tensor = preprocess_llama11B(image_tensor_ori, self.processor.image_processor).unsqueeze(0).unsqueeze(0)
            
            torch.cuda.empty_cache()

        # Convert the perturbed tensor to an image and save it
        image_perturbed = image_tensor_ori.cpu().detach().numpy()
        image_perturbed = np.clip(image_perturbed, 0, 1)
        image_perturbed = image_perturbed.transpose(1, 2, 0).astype('float32')

        return image_perturbed, image_tensor
    
    def eval_llama_11B_model(self, item, add_prompt=None, args=None, gt=None):
        
        self.image_file = item["image_path"]
        self.qs = item["question"]

        if not add_prompt is None:
            self.qs += add_prompt
            
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.qs}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)  

        # Ensure model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        
        image_perturbed, image_perturbed_processed = self.PGD_attack_llama_11B(
            inputs = inputs,
            gt = gt,
            task = item["question_type"],
            iterations=args.num_iter,
            alpha=args.alpha/255,
            epsilon=args.epsilon/255,
            args=args)
        
        inputs["pixel_values"] = image_perturbed_processed
            
        if self.save_image:
            self.save_purterbed_image(image_perturbed, args)

        # Generate outputs with the perturbed image
        with torch.inference_mode():
            outputs = self.model.generate(**inputs,
                                    do_sample=True if args.temperature > 0 else False,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    num_beams=args.num_beams,
                                    # no_repeat_ngram_size=3,
                                    max_new_tokens=1024,
                                    use_cache=True,
                                      )
        
        outputs = self.processor.decode(outputs[0]).replace(input_text, "").replace("<|eot_id|>","").strip()
        
        return outputs
           
    def eval_llava_model(self, item, add_prompt=None, args=None, gt=None):

        self.image_file = item["image_path"]
        self.qs = item["question"]

        if not add_prompt is None:
            self.qs += add_prompt

        if IMAGE_PLACEHOLDER in self.qs:
            if self.model.config.mm_use_im_start_end:
                self.qs = re.sub(IMAGE_PLACEHOLDER, self.image_token_se, self.qs)
            else:
                self.qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, self.qs)
        else:
            if self.model.config.mm_use_im_start_end:
                self.qs = self.image_token_se + "\n" + self.qs
            else:
                self.qs = DEFAULT_IMAGE_TOKEN + "\n" + self.qs
                
        image_path = os.path.join(args.image_folder, self.image_file)
        image = Image.open(image_path).convert('RGB')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Prepare input_ids without gradients
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids.requires_grad_(False)     


        # Ensure model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()


        image_perturbed, image_perturbed_processed = self.PGD_attack_llava(
            input_ids = input_ids,
            gt = gt,
            task = item["question_type"],
            iterations=args.num_iter,
            alpha=args.alpha/255,
            epsilon=args.epsilon/255,
            args=args)
            
        if self.save_image:
            self.save_purterbed_image(image_perturbed, args)

        # Generate outputs with the perturbed image
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
        """ Evaluate the dolphins model on a given item. """
        self.image_file = item["image_path"]
        question = item["question"]
        self.qs = question

        if add_prompt:
            question += add_prompt

        # Generate prompt with your predefined method for the 'dolphins' model
        vision_x, inputs = dolphins.get_model_inputs(os.path.join(args.image_folder, self.image_file), question, self.model, self.image_processor, self.tokenizer)

        # Ensure model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        
        image_perturbed, image_perturbed_processed  = self.PGD_attack_dolphins(
            inputs = inputs,
            gt = gt,
            task = item["question_type"],
            iterations=args.num_iter,
            alpha=args.alpha/255,
            epsilon=args.epsilon/255,
            args=args)
        
        if self.save_image:
            self.save_purterbed_image(image_perturbed, args)
        
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
        
        output_ids = self.model.generate(
            vision_x=image_perturbed_processed,
            lang_x=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].cuda(),
            num_beams=3,
            **generation_kwargs,
        )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.split("GPT:")[1].strip()
        return outputs
    
    def eval_EM_VLM4AD_model(self, item, add_prompt=None, ignore_type=False, args=None, gt=None):
        self.image_file = item["image_path"]
        question = item["question"]
            
        if add_prompt:
            question += add_prompt
            self.qs = question
            
        # Ensure model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        
        image_perturbed, image_perturbed_processed  = self.PGD_attack_EM_VLM4AD(
            question = question,
            gt = gt,
            task = item["question_type"],
            iterations=args.num_iter,
            alpha=args.alpha/255,
            epsilon=args.epsilon/255,
            args=args)
            
        if self.save_image:
            self.save_purterbed_image(image_perturbed, args)

        question_encodings = self.tokenizer(question, padding=True, return_tensors="pt").input_ids.to(self.device)

        if ignore_type:
            text_output = EM_VLM4AD.val_model_without_preprocess(self.model, self.tokenizer, question, image_perturbed_processed)[0]
        else:
            if item['question_type'] == 'open-ended':
                text_output = EM_VLM4AD.val_model_without_preprocess(self.model, self.tokenizer, question, image_perturbed_processed)[0]
            else:
                # embed_output = EM_VLM4AD.causual_model_without_preprocess(self.model, self.tokenizer, question, image_perturbed_processed)
                embed_output = self.model(question_encodings, image_perturbed_processed)
                if item['question_type'] == 'multi-choice':
                    label_pool = ['A', 'B', 'C', 'D']
                    token_id_A = self.tokenizer.encode("A", add_special_tokens=False)
                    token_id_B = self.tokenizer.encode("B", add_special_tokens=False)
                    token_id_C = self.tokenizer.encode("C", add_special_tokens=False)
                    token_id_D = self.tokenizer.encode("D", add_special_tokens=False)
                    assert len(token_id_A) == 1 and len(token_id_B) == 1 and len(token_id_C) == 1 and len(token_id_D) == 1
                    token_id_A = token_id_A[0]
                    token_id_B = token_id_B[0]
                    token_id_C = token_id_C[0]
                    token_id_D = token_id_D[0]
                    logits_sft = F.softmax(embed_output.logits[:,-1,:], dim=-1)
                    logits_A_B_C_D = logits_sft[:, [token_id_A, token_id_B, token_id_C, token_id_D]].cpu()
                    pred_index = logits_A_B_C_D.argmax(dim=-1).item()
                    text_output = label_pool[pred_index]
                elif item['question_type'] == 'yes-or-no':
                    label_pool = ['Yes', 'No']
                    token_id_A = self.tokenizer.encode("yes", add_special_tokens=False)
                    token_id_B = self.tokenizer.encode("no", add_special_tokens=False)
                    assert len(token_id_A) == 1 and len(token_id_B) == 1 
                    token_id_A = token_id_A[0]
                    token_id_B = token_id_B[0]
                    logits_sft = F.softmax(embed_output.logits[:,-1,:], dim=-1)
                    logits_A_B = logits_sft[:, [token_id_A, token_id_B]].cpu()
                    pred_index = logits_A_B.argmax(dim=-1).item()
                    text_output = label_pool[pred_index]
                    
        return text_output
    
    def eval_blip2(self, item, add_prompt=None, args=None, gt=None):

        self.image_file = item["image_path"]
        self.qs = item["question"]

        if not add_prompt is None:
            self.qs += add_prompt
            
        # Ensure model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        
        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        inputs = self.processor(image, self.qs, return_tensors="pt").to(self.device)
        # image_test = inputs['pixel_values'] # not used
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        image_perturbed, image_perturbed_processed  = self.PGD_attack_drivelm_agent(
            question = input_ids,
            gt = gt,
            task = item["question_type"],
            iterations=args.num_iter,
            alpha=args.alpha/255,
            epsilon=args.epsilon/255,
            args=args)
            
        if self.save_image:
            self.save_purterbed_image(image_perturbed, args)    

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

        if not add_prompt is None:
            self.qs += add_prompt
        
        # Ensure model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        
        prompt = [llama.format_prompt(self.qs)]
        
        image_perturbed, image_perturbed_processed  = self.PGD_attack_llama_adapter(
            question = prompt,
            gt = gt,
            task = item["question_type"],
            iterations=args.num_iter,
            alpha=args.alpha/255,
            epsilon=args.epsilon/255,
            args=args)
        
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
                if not dataset_filtering_func(item):
                    continue
                if prefix is not None:
                    if isinstance(prefix, dict):
                        prefix_i = prefix[str(item["question_id"])]
                    elif isinstance(prefix, str):
                        prefix_i = prefix
                    else:
                        raise ValueError("Invalid prefix type")
                else: 
                    prefix_i = None
                
                
                item, add_prompt = self.process_input(item, prefix=prefix_i, suffix=suffix, question=question)
                response = self.eval_llava_model(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])

                item["question"] = self.qs
                ans_dict = {"llava_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
                
        elif 'Dolphins' in self.model_name:
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                if prefix is not None:
                    if isinstance(prefix, dict):
                        prefix_i = prefix[str(item["question_id"])]
                    elif isinstance(prefix, str):
                        prefix_i = prefix
                    else:
                        raise ValueError("Invalid prefix type")
                else: 
                    prefix_i = None
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
                if not dataset_filtering_func(item):
                    continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_EM_VLM4AD_model(item, add_prompt=add_prompt, ignore_type=ignore_type, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"EM_VLM4AD_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)    
                
        elif "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                
                response = self.eval_blip2(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"drivelm_agent_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item) 
                
        elif "llama_adapter" in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                
                response = self.eval_llama_adapter(item, add_prompt=add_prompt, args=args, gt=gts[item["question_id"]]['answer'])
                item["question"] = self.qs
                ans_dict = {"llama_adapter_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item) 
        
        elif "Llama-3.2-11B-Vision" in self.model_name:
            self.init_llama_11B(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
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
            
            
