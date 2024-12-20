from evaluator.BaseEvaluator import BaseEvaluator
from tqdm import tqdm
import json


class FairnessEvaluator(BaseEvaluator):
    def __init__(self, args):
        super(FairnessEvaluator, self).__init__(args)
        if args.pattern_file:
            with open(args.pattern_file, 'r') as f:
                self.prefixes = json.load(f)
        self.subtype = args.subtype

    def evaluate_base(self, args=None):
        print("Evaluating base case.")
        self.answer_qa(args=args)

    def evaluate_driver(self, mode='race', args=None):
        if mode == "race":
            print(f"Evaluating driver by race with subtype {self.subtype}.")
            if self.subtype in self.prefixes[0]["Driver"]["race"]:
                prefix = self.prefixes[0]["Driver"]["race"][self.subtype]
                print(f"Prefix: {prefix}")
                self.answer_qa(prefix=prefix, args=args)
        elif mode == "gender":
            print(f"Evaluating driver by gender with subtype {self.subtype}.")
            if self.subtype in self.prefixes[0]["Driver"]["gender"]:
                prefix = self.prefixes[0]["Driver"]["gender"][self.subtype]
                print(f"Prefix: {prefix}")
                self.answer_qa(prefix=prefix, args=args)
        elif mode == "age":
            print(f"Evaluating driver by age with subtype {self.subtype}.")
            if self.subtype in self.prefixes[0]["Driver"]["age"]:
                prefix = self.prefixes[0]["Driver"]["age"][self.subtype]
                print(f"Prefix: {prefix}")
                self.answer_qa(prefix=prefix, args=args)

    def evaluate_vehicle(self, mode='type', args=None):
        if mode == "type":
            print(f"Evaluating vehicle by type with subtype {self.subtype}.")
            if self.subtype in self.prefixes[1]["Vehicle"]["vehicle_type"]:
                prefix = self.prefixes[1]["Vehicle"]["vehicle_type"][self.subtype]
                print(f"Prefix: {prefix}")
                self.answer_qa(prefix=prefix, args=args)
        elif mode == "color":
            print(f"Evaluating vehicle by color with subtype {self.subtype}.")
            if self.subtype in self.prefixes[1]["Vehicle"]["vehicle_color"]:
                prefix = self.prefixes[1]["Vehicle"]["vehicle_color"][self.subtype]
                print(f"Prefix: {prefix}")
                self.answer_qa(prefix=prefix, args=args)
        elif mode == "brand":
            print(f"Evaluating vehicle by color with subtype {self.subtype}.")
            if self.subtype in self.prefixes[1]["Vehicle"]["vehicle_brand"]:
                prefix = self.prefixes[1]["Vehicle"]["vehicle_brand"][self.subtype]
                print(f"Prefix: {prefix}")
                self.answer_qa(prefix=prefix, args=args)
    
    def evaluate_scene(self, args=None):
        print("Evaluating Fairness Scene.")
        self.answer_qa(args=args)

        
   