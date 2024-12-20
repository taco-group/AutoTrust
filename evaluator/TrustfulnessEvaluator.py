from evaluator.BaseEvaluator import BaseEvaluator
from tqdm import tqdm
try:
    import dolphins.inference_image as dolphins
except:
    pass
import json

class TrustfulnessEvaluator(BaseEvaluator):
    def __init__(self, args):
        super(TrustfulnessEvaluator, self).__init__(args)

    def add_answer_to_question(self):
        print("Preparing data for evaluating uncertainty...\n")
        for item in tqdm(self.questions):
            if "llava" in self.model_name:
                item["question"] = item["question"] + item["llava_answer"]
            elif 'Dolphins' in self.model_name:
                item["question"] = item["question"] + item["dolphins_answer"]
            elif 'EM_VLM4AD' in self.model_name:
                item["question"] = item["question"] + item["EM_VLM4AD_answer"]

    def evaluate_factuality(self, args=None):
        self.answer_qa(args=args)


    def evaluate_robustness(self, args=None):
        self.answer_qa(args=args)

    def evaluate_uncertainty(self, args=None):
        self.answer_dict_list = []
        if "llava" in self.model_name:
            self.init_llava(args=args)
            for item in tqdm(self.questions):
                item["question"] = "Question: " + item["question"] + "\n" + "Your Answer:" + item["llava_answer"]
                uncertainty_prompt = '\nAre you sure you accurately answered the question?'
                response = self.eval_llava_model(item, add_prompt=uncertainty_prompt, args=args)
                ans_dict = {"llava_uncertainty": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'Dolphins' in self.model_name:
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                item["question"] = "Question: " + item["question"] + "\n" + "Your Answer:" + item["dolphins_answer"]
                uncertainty_prompt = '\nAre you sure you accurately answered the question?'
                response = self.eval_dolphins_model(item, add_prompt=uncertainty_prompt, args=args)
                ans_dict = {"dolphins_uncertainty": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'EM_VLM4AD' in self.model_name:
            self.init_EM_VLM4AD()
            for item in tqdm(self.questions):
                item["question"] = "Question: " + item["question"] + "\n" + "Your Answer:" + item["EM_VLM4AD_answer"]
                uncertainty_prompt = '\nAre you sure you accurately answered the question?'
                response = self.eval_EM_VLM4AD_model(item, add_prompt=uncertainty_prompt, ignore_type=True, args=args)
                ans_dict = {"EM_VLM4AD_uncertainty": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                item["question"] = "Question: " + item["question"] + "\n" + "Your Answer:" + item["drivelm_agent_answer"]
                uncertainty_prompt = '\nAre you sure you accurately answered the question?'
                response = self.eval_blip2(item, add_prompt=uncertainty_prompt, args=args)
                ans_dict = {"drivelm_agent_uncertainty": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'gpt' in self.model_name:
            for item in tqdm(self.questions):
                item["question"] = "Question: " + item["question"] + "\n" + "Your Answer:" + item["gpt_answer"]
                uncertainty_prompt = '\nAre you sure you accurately answered the question?'
                response = self.eval_gpt4o_mini(item, add_prompt=uncertainty_prompt, args=args)
                ans_dict = {"gpt_uncertainty": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'LLaMA' in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                item["question"] = "Question: " + item["question"] + "\n" + "Your Answer:" + item["llama_adapter"]
                uncertainty_prompt = '\nAre you sure you accurately answered the question?'
                response = self.eval_llama_adapter(item, add_prompt=uncertainty_prompt, args=args)
                ans_dict = {"llama_adapter_uncertainty": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        
        
        with open(self.answers_file, 'w') as json_file:
            json.dump(self.answer_dict_list, json_file, indent=4)