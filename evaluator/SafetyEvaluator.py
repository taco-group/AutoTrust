from evaluator.BaseEvaluator import BaseEvaluator
from tqdm import tqdm
from evaluator.utils import list2dict
import json
import os
import random
from preprocess.safety.misinstruction import set_seed, instruction_dict
from preprocess.safety.misinformation import set_seed, generate_misinfo_prefix




class SafetyEvaluator(BaseEvaluator):
    
    def __init__(self, args):
        super().__init__(args)
            
    def get_prefix(self, prefix, item):
        if prefix is not None:
            if isinstance(prefix, dict):
                try:
                    return prefix[int(item["question_id"])]
                except:
                    return prefix[str(item["question_id"])]
            elif isinstance(prefix, str):
                return prefix
            elif isinstance(prefix, list):
                return random.choice(prefix)
            else:
                raise ValueError("Invalid prefix type")
    
    
    def process_input(self, item, prefix=None, suffix=None, question=None, ignore_type=False, perturbed=False):
        if not question is None:
            item["question"] =  question
        if not prefix is None:
            item["question"] = prefix + " " + item["question"]
        if not suffix is None:
            item["question"] = item["question"] + " " + suffix

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
            
        if perturbed:
            item["image_path"] = os.path.join(
                f"{item['image_path'].split('/')[0]}_perturbed_16_1.0_30_llama_11B",
                *item['image_path'].split('/')[1:]
            )

        return item, add_prompt
    
    def evaluate_black_box_attack(self, args=None):
        '''
        Black-box attack
        '''
        dataset_filtering_func = lambda x: x["question_type"] != "open-ended"
        self.answer_qa(args=args, dataset_filtering_func=dataset_filtering_func, perturbed=True)
            
            
    def evaluate_misinstruction(self, args=None):
        '''
        Consistency against Misinstruction
        
        Supports close-end question only
        '''
        prefix = None
        dataset_filtering_func = lambda x: x["question_type"] != "open-ended"
            
        # set random seed for matching misinstruction with sentences.
        set_seed(args.seed)
        
        qs_ctype_path = f"preprocess/safety/task_cognitive_types/{args.dataset}.json"
        assert os.path.exists(qs_ctype_path), f"{qs_ctype_path} not found."
        with open(qs_ctype_path, 'r') as file:
            qs_ctype = json.load(file)
        
        inst_dict = dict()
        for qid, ctype in qs_ctype.items():
            misinstructions = instruction_dict[ctype]
            inst_dict[qid] = random.choice(misinstructions)
        prefix = inst_dict
        self.answer_qa(args=args, prefix=prefix, dataset_filtering_func=dataset_filtering_func)
        
        
    def evaluate_misinformation(self, args=None):
        '''
        Contextual Misinformation
            
        Supports close-end question only
        '''
        
        # set random seed for matching misinstruction with sentences.
        set_seed(args.seed)
        
        prefix = None
        dataset_filtering_func = lambda x: x["question_type"] != "open-ended"
        prefix = generate_misinfo_prefix(args.dataset)
        self.answer_qa(args=args, prefix=prefix, dataset_filtering_func=dataset_filtering_func)
            

    def answer_qa(self, prefix=None, suffix=None, question=None, ignore_type=False, args=None, perturbed=False ,dataset_filtering_func=lambda x: True):
        """

        Args:
            args (optional): Defaults to None.
            prefix (dict or string, optional): Prefix for each question or for all questions. Defaults to None.
            dataset_filtering_func (function, optional): Filter function for each question. Defaults to lambda x: True.

        Return: None
        """
        
        self.answer_dict_list = []
        
        if "llava" in self.model_name:
            self.init_llava(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                px = self.get_prefix(prefix, item)
                item, add_prompt = self.process_input(item, prefix=px, suffix=suffix, question=question, ignore_type=ignore_type, perturbed=perturbed)
                response = self.eval_llava_model(item, add_prompt=add_prompt, args=args)
                item["question"] = self.qs
                ans_dict = {"llava_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'Dolphins' in self.model_name:
            import dolphins.inference_image as dolphins
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                px = self.get_prefix(prefix, item)
                item, add_prompt = self.process_input(item, prefix=px, suffix=suffix, question=question, ignore_type=ignore_type, perturbed=perturbed)
                response = self.eval_dolphins_model(item, add_prompt=add_prompt, args=args)
                item["question"] = self.qs
                ans_dict = {"dolphins_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
                
        elif 'EM_VLM4AD' in self.model_name:
            self.init_EM_VLM4AD()
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                px = self.get_prefix(prefix, item)
                item, add_prompt = self.process_input(item, prefix=px, suffix=suffix, question=question, ignore_type=ignore_type, perturbed=perturbed)
                response = self.eval_EM_VLM4AD_model(item, add_prompt=add_prompt, ignore_type=ignore_type, args=args)
                item["question"] = self.qs
                ans_dict = {"EM_VLM4AD_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                px = self.get_prefix(prefix, item)
                item, add_prompt = self.process_input(item, prefix=px, suffix=suffix, question=question, ignore_type=ignore_type, perturbed=perturbed)
                response = self.eval_blip2(item, add_prompt=add_prompt, args=args)
                item["question"] = self.qs
                ans_dict = {"drivelm_agent_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'gpt' in self.model_name:
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                px = self.get_prefix(prefix, item)
                item, add_prompt = self.process_input(item, prefix=px, suffix=suffix, question=question, ignore_type=ignore_type, perturbed=perturbed)
                response = self.eval_gpt4o_mini(item, add_prompt=add_prompt, args=args)
                item["question"] = self.qs
                ans_dict = {"gpt_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'llama_adapter' in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                if not dataset_filtering_func(item):
                    continue
                px = self.get_prefix(prefix, item)
                item, add_prompt = self.process_input(item, prefix=px, suffix=suffix, question=question, ignore_type=ignore_type, perturbed=perturbed)
                response = self.eval_llama_adapter(item, add_prompt=add_prompt, args=args)
                item["question"] = self.qs
                ans_dict = {"llama_adapter_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        os.makedirs(os.path.dirname(self.answers_file), exist_ok=True)      
        with open(self.answers_file, "w") as json_file:
            json.dump(self.answer_dict_list, json_file, indent=4)
