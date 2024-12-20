import torch
import os
import json
from tqdm import tqdm
import re
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
try:
    import dolphins.inference_image as dolphins
except:
    pass
import EM_VLM4AD.eval_single as EM_VLM4AD
from utils import query_gpt4
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import PeftModel
from DriveLM import llama 
import cv2
import torchvision.transforms as transforms

from concurrent.futures import ThreadPoolExecutor, as_completed


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class BaseEvaluator:
    def __init__(self, args):
        super(BaseEvaluator, self).__init__()
        self.model_path = os.path.expanduser(args.model_path)
        self.model_name = get_model_name_from_path(self.model_path)
        with open(args.question_file, 'r') as file:
            self.questions = json.load(file)
        self.answers_file = os.path.expanduser(args.answers_file)
        self.model_base = args.model_base
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


    def process_input(self, item, prefix=None, suffix=None, question=None, ignore_type=False,):
        if not question is None:
            item["question"] =  question
        if not prefix is None:
            item["question"] = prefix + item["question"]
        if not suffix is None:
            item["question"] = item["question"] + suffix

        if not ignore_type:
            # add templates for anwsering closed-ended questions
            if item["question_type"] == "multi-choice":
                add_prompt = '\n' + "Answer with the option's letter from the given choices directly."
            elif item["question_type"] == "yes-or-no":
                add_prompt = '\n' + "Answer with Yes or No directly."
            else:
                add_prompt = None
        else:
            add_prompt = None

        return item, add_prompt

    def init_llava(self, args=None):
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name)
        # if args.load_peft is not None:
        #     self.model.load_adapter(args.load_peft)
        #     print(f"Loaded adapter!")

        self.image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if args.conv_mode is not None and self.conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    self.conv_mode, args.conv_mode, args.conv_mode
                )
            )
            self.conv_mode = args.conv_mode

    def init_blip2(self, args=None):
        disable_torch_init()
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.bfloat16,
            device_map=self.device)
        
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        if args.load_peft is not None:
            # self.model.load_adapter(args.load_peft)
            self.model = PeftModel.from_pretrained(
                self.model, 
                args.load_peft, 
                use_auth_token="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"
                )
            print(f"Loaded adapter from {args.load_peft}")
            
    def init_llama_adapter(self, args=None):
        self.model, self.preprocess = llama.load(args.checkpoint, args.model_path, llama_type="7B", device=self.device)
        print('4444444', self.model)
        self.model.eval()
        print(f"Loaded llama_adapter checkpoint from {args.checkpoint}")


    def eval_llava_model(self, item, add_prompt=None, args=None):

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

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def eval_dolphins_model(self, item, add_prompt=None,args=None):
        """ Evaluate the dolphins model on a given item. """
        image_path = item["image_path"]
        question = item["question"]

        if add_prompt:
            question += add_prompt
            self.qs = question

        # Generate prompt with your predefined method for the 'dolphins' model
        vision_x, inputs = dolphins.get_model_inputs(os.path.join(args.image_folder, image_path), question, self.model, self.image_processor, self.tokenizer)

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
            vision_x=vision_x.half().to(self.device),
            lang_x=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            num_beams=3,
            **generation_kwargs,
        )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.split("GPT:")[1].strip()
        return outputs
    
    
    def init_EM_VLM4AD(self):
        # inferencing on single batch for now
        config = {
        'batch_size': 1,
        'epochs': 15,
        'gpa_hidden_size': 128,
        'freeze_lm': False,
        'lm': 'T5-Large',
        'lora': False,
        'lora_dim': 64,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'max_len': 512,
        'num_workers': 0,
        'model_name': 'T5-Large'
        }
        self.model, self.tokenizer, self.image_processor = EM_VLM4AD.init_model(config)
        
    def eval_EM_VLM4AD_model(self, item, add_prompt=None, ignore_type=False, args=None):
        image_path = item["image_path"]
        question = item["question"]
            
        if add_prompt:
            question += add_prompt
            self.qs = question

        if ignore_type:
            text_output = EM_VLM4AD.val_model(self.model.to(self.device), self.tokenizer, self.image_processor, question, os.path.join(args.image_folder, image_path))[0]
        else:
            if item['question_type'] == 'open-ended':
                text_output = EM_VLM4AD.val_model(self.model.to(self.device), self.tokenizer, self.image_processor, question, os.path.join(args.image_folder, image_path))[0]
            else:
                embed_output = EM_VLM4AD.causual_model(self.model.to(self.device), self.tokenizer, self.image_processor, question, os.path.join(args.image_folder, image_path))
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
                    logits_sft = F.softmax(embed_output[-1], dim=-1)
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
                    logits_sft = F.softmax(embed_output[-1], dim=-1)
                    logits_A_B = logits_sft[:, [token_id_A, token_id_B]].cpu()
                    pred_index = logits_A_B.argmax(dim=-1).item()
                    text_output = label_pool[pred_index]
        return text_output
    
    def eval_gpt4o_mini(self, item, add_prompt=None, args=None):
        self.image_file = item["image_path"]
        self.qs = item["question"]

        if not add_prompt is None:
            self.qs += add_prompt

        image_path =  os.path.join(args.image_folder, self.image_file)
        outputs = query_gpt4(
            self.qs, api_key=args.api_key, 
            image_path=image_path
            )
        return outputs
    
    def eval_blip2(self, item, add_prompt=None, args=None):

        self.image_file = item["image_path"]
        self.qs = item["question"]

        if not add_prompt is None:
            self.qs += add_prompt

        image = Image.open(os.path.join(args.image_folder, self.image_file)).convert('RGB')
        inputs = self.processor(image, self.qs, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                temperature=args.temperature,
                )
        outputs = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return outputs
    
    def eval_llama_adapter(self, item, add_prompt=None, args=None):
        self.image_file = item["image_path"]
        self.qs = item["question"]

        if not add_prompt is None:
            self.qs += add_prompt
        image = cv2.imread(os.path.join(args.image_folder, self.image_file))
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0).unsqueeze(0)
        prompt = [llama.format_prompt(self.qs)]
        outputs = self.model.generate(image.to(self.device), prompt, temperature=0.2, top_p=0.1)
        return outputs[0]
    
    def gpt_process_and_eval(self, item, prefix='', suffix='', question='', ignore_type='', args=None):
        item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
        response = self.eval_gpt4o_mini(item, add_prompt=add_prompt, args=args)
        ans_dict = {"gpt_answer": response}
        answered_item = {**item, **ans_dict}
        self.answer_dict_list.append(answered_item)
    
    
    def answer_qa(self, prefix=None, suffix=None, question=None, ignore_type=False, args=None):
        self.answer_dict_list = []
        if "llava" in self.model_name:
            self.init_llava(args=args)
            for item in tqdm(self.questions):
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llava_model(item, add_prompt=add_prompt, args=args)
                item["question"] = self.qs
                ans_dict = {"llava_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'Dolphins' in self.model_name:
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_dolphins_model(item, add_prompt=add_prompt, args=args)
                # item["question"] = self.qs
                ans_dict = {"dolphins_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'EM_VLM4AD' in self.model_name:
            self.init_EM_VLM4AD()
            for item in tqdm(self.questions):
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_EM_VLM4AD_model(item, add_prompt=add_prompt, ignore_type=ignore_type, args=args)
                ans_dict = {"EM_VLM4AD_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        if "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_blip2(item, add_prompt=add_prompt, args=args)
                ans_dict = {"drivelm_agent_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'gpt' in self.model_name:
            for item in tqdm(self.questions):
                # item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                # response = self.eval_gpt4o_mini(item, add_prompt=add_prompt, args=args)
                # ans_dict = {"gpt_answer": response}
                # answered_item = {**item, **ans_dict}
                # self.answer_dict_list.append(answered_item)
                self.gpt_process_and_eval(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type, args=args)
        elif "LLaMA" in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llama_adapter(item, add_prompt=add_prompt, args=args)
                print(response)
                ans_dict = {"llama_adapter": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif "gpt-multithread" in self.model_name:
            with ThreadPoolExecutor(max_workers=60) as executor:
                future_to_question = {executor.submit(self.gpt_process_and_eval, item, prefix, suffix, question, ignore_type, args): item for item in self.questions}
                for future in as_completed(future_to_question):
                    answered_item = future.result()
                    self.answer_dict_list.append(answered_item)
            

        
        with open(self.answers_file, 'w') as json_file:
            json.dump(self.answer_dict_list, json_file, indent=4)