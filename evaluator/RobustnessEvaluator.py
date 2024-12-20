from evaluator.BaseEvaluator import BaseEvaluator
from tqdm import tqdm
import dolphins.inference_image as dolphins
import json
import os

class RobustnessEvaluator(BaseEvaluator):
    def __init__(self, args):
        super(RobustnessEvaluator, self).__init__(args)

    def add_answer_to_question(self):
        print("Preparing data for evaluating uncertainty...\n")
        for item in tqdm(self.questions):
            if "llava" in self.model_name:
                item["question"] = item["question"] + item["llava_answer"]
            elif 'Dolphins' in self.model_name:
                item["question"] = item["question"] + item["dolphins_answer"]
            elif 'EM_VLM4AD' in self.model_name:
                item["question"] = item["question"] + item["EM_VLM4AD_answer"]


    # def process_input(self, item, prefix=None, suffix=None, question=None, ignore_type=False,):
    #     if not question is None:
    #         item["question"] =  question
    #     if not prefix is None:
    #         item["question"] = prefix + item["question"]
    #     if not suffix is None:
    #         item["question"] = item["question"] + suffix

    #     if not ignore_type:
    #         # add templates for anwsering closed-ended questions
    #         if item["question_type"] == "multi-choice":
    #             add_prompt = '\n' + "Answer with the option's letter from the given choices directly."
    #         elif item["question_type"] == "yes-or-no":
    #             add_prompt = '\n' + "Answer with Yes or No directly."
    #         else:
    #             add_prompt = None
    #     else:
    #         add_prompt = None
        
    #     # item['image_path'] = os.path.basename(item['image_path'])

    #     return item, add_prompt

    # def evaluate_robustness(self, args=None):
    #     self.answer_qa(args=args)

    def evaluate_robustness_withAbs(self, prefix=None, suffix=None, question=None, ignore_type=True, args=None): #jinlong: ignore_type=True
        self.answer_dict_list = []
        # self.ood_prompt = "If you have not encountered relevant data during training, you can decline to answer or output 'I don't know'. Answer in English."
        self.ood_prompt = "If you have not encountered relevant data during training, you can decline to answer or output 'I don't know'. Please answer the above question in English."

        if "llava" in self.model_name:
            self.init_llava(args=args)
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])
                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")

                # response = self.eval_llava_model(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llava_model(item, add_prompt=add_prompt, args=args)
                ans_dict = {"llava_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'Dolphins' in self.model_name:
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")



                # response = self.eval_dolphins_model(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_dolphins_model(item, add_prompt=add_prompt, args=args)

                ans_dict = {"dolphins_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'EM_VLM4AD' in self.model_name:
            self.init_EM_VLM4AD()
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")
                response = self.eval_EM_VLM4AD_model(item,ignore_type=ignore_type,args=args)
                ans_dict = {"EM_VLM4AD_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")
                # response = self.eval_blip2(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_blip2(item, add_prompt=add_prompt, args=args)

                ans_dict = {"drivelm_agent_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'gpt' in self.model_name:
            for item in tqdm(self.questions):



                if item["question_type"] == "open-ended":

                    print("skip the ", item["question_type"]+" in " + item['image_path'])

                    continue

                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")

                # response = self.eval_gpt4o_mini(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_gpt4o_mini(item, add_prompt=add_prompt, args=args)



                ans_dict = {"gpt_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'LLaMA' in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")

                # response = self.eval_llama_adapter(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llama_adapter(item, add_prompt=add_prompt, args=args)   

                ans_dict = {"llama_adapter_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        
        
        with open(self.answers_file, 'w', encoding='utf-8') as json_file:

            json.dump(self.answer_dict_list, json_file, ensure_ascii=False, indent=4)




    def evaluate_robustness_acc(self, prefix=None, suffix=None, question=None, ignore_type=False, args=None):
        self.answer_dict_list = []
        # self.ood_prompt = "Answer in English."
        self.ood_prompt = " "
        
        if "llava" in self.model_name:
            self.init_llava(args=args)
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])
                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")

                # response = self.eval_llava_model(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llava_model(item, add_prompt=add_prompt, args=args)
                ans_dict = {"llava_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'Dolphins' in self.model_name:
            self.model, self.image_processor, self.tokenizer = dolphins.load_pretrained_model()
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")



                # response = self.eval_dolphins_model(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_dolphins_model(item, add_prompt=add_prompt, args=args)

                ans_dict = {"dolphins_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'EM_VLM4AD' in self.model_name:
            self.init_EM_VLM4AD()
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")
                response = self.eval_EM_VLM4AD_model(item,args=args)
                ans_dict = {"EM_VLM4AD_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif "drivelm-agent" in self.model_name:
            self.init_blip2(args=args)
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")
                # response = self.eval_blip2(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_blip2(item, add_prompt=add_prompt, args=args)

                ans_dict = {"drivelm_agent_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)

        elif 'gpt' in self.model_name:
            for item in tqdm(self.questions):



                if item["question_type"] == "open-ended":

                    print("skip the ", item["question_type"]+" in " + item['image_path'])

                    continue

                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")

                # response = self.eval_gpt4o_mini(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_gpt4o_mini(item, add_prompt=add_prompt, args=args)

                ans_dict = {"gpt_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        elif 'LLaMA' in self.model_name:
            self.init_llama_adapter(args=args)
            for item in tqdm(self.questions):
                item["question"] = item["question"] + "\n" + self.ood_prompt
                if 'CAM_FRONT' in item['image_path']:
                    item['image_path'] = os.path.basename(item['image_path'])

                elif 'Cityscapes' in item['image_path']:
                    item['image_path'] = item['image_path'].lstrip("./")

                # response = self.eval_llama_adapter(item, args=args)
                item, add_prompt = self.process_input(item, prefix=prefix, suffix=suffix, question=question, ignore_type=ignore_type)
                response = self.eval_llama_adapter(item, add_prompt=add_prompt, args=args)   
                             
                ans_dict = {"llama_adapter_ood_answer": response}
                answered_item = {**item, **ans_dict}
                self.answer_dict_list.append(answered_item)
        

        with open(self.answers_file, 'w', encoding='utf-8') as json_file:

            json.dump(self.answer_dict_list, json_file, ensure_ascii=False, indent=4)


