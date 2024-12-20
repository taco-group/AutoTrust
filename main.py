import argparse
from evaluator.BaseEvaluator import BaseEvaluator
from evaluator.TrustfulnessEvaluator import TrustfulnessEvaluator
from evaluator.PrivacyEvaluator import PrivacyEvaluator
from evaluator.SafetyEvaluator import SafetyEvaluator
from evaluator.AdvEvaluator import AdvEvaluator
from evaluator.FairnessEvaluator import FairnessEvaluator
from evaluator.RobustnessEvaluator import RobustnessEvaluator

def main(args):
    if args.task in ["factuality", "uncertainty"]:
        evaluator = TrustfulnessEvaluator(args)
        if args.task == "factuality":
            evaluator.evaluate_factuality(args=args)  
        elif args.task == "uncertainty":
            evaluator.evaluate_uncertainty(args=args)

    if args.task in ["identity_individual", "identity_vehicle", "location"]:
        evaluator = PrivacyEvaluator(args)
        if args.task == "identity_individual":
            evaluator.evaluate(args=args)  
        elif args.task == "identity_vehicle":
            evaluator.evaluate(args=args)
        elif args.task == "location":
            evaluator.evaluate(args=args)
            
    if args.task in ["misinstruction", "misinformation", "black_box_attack"]:
        evaluator = SafetyEvaluator(args)
        if args.task == "misinstruction":
            evaluator.evaluate_misinstruction(args=args)  
        elif args.task == "misinformation":
            evaluator.evaluate_misinformation(args=args)
        elif args.task == "black_box_attack":
            evaluator.evaluate_black_box_attack(args=args)
            
    if args.task in ["white_box_attack"]:
        evaluator = AdvEvaluator(args, save_image=True)
        evaluator.evaluate_white_box_attack(args=args)

    if args.task in ["driver_race", "driver_gender", "driver_age", "vehicle_type", "vehicle_color", "vehicle_brand", "None"]:
        evaluator = FairnessEvaluator(args)
        if args.task == "driver_race":
            evaluator.evaluate_driver(mode='race', args=args)
        elif args.task == "driver_gender":
            evaluator.evaluate_driver(mode='gender', args=args)
        elif args.task == "driver_age":
            evaluator.evaluate_driver(mode='age', args=args) 
        elif args.task == "vehicle_type":
            evaluator.evaluate_vehicle(mode='type', args=args)
        elif args.task == "vehicle_color":
            evaluator.evaluate_vehicle(mode='color', args=args)
        elif args.task == "vehicle_brand":
            evaluator.evaluate_vehicle(mode='brand', args=args)
        elif args.task == "None":
            evaluator.evaluate_base(args=args)

    if args.task in ["scene"]:
        evaluator = FairnessEvaluator(args)
        evaluator.evaluate_scene(args=args)

    if args.task in ["robustness_acc", "robustness_abs"]:
        if args.task == "robustness_abs":
            evaluator = RobustnessEvaluator(args)
            evaluator.evaluate_robustness_withAbs(args=args)
        elif args.task == "robustness_acc":
            evaluator = RobustnessEvaluator(args)
            evaluator.evaluate_robustness_acc(args=args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-peft", type=str, default='shuoxing/drivelm-blip2-lora-ckpt')
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument('--llama_dir', type=str, default="/path/to/llama_model_weights", help='path to llama model weights')
    parser.add_argument('--checkpoint', type=str, default="/path/to/pre-trained/checkpoint.pth", help='path to pre-trained checkpoint')
    parser.add_argument("--pattern-file", type=str, default=None)
    parser.add_argument("--subtype", type=str, default=None)
    parser.add_argument("--gt-file", type=str, default="gt.json", help='ground truth file')
    parser.add_argument("--dataset", type=str, default="DriveLM", help='dataset name') #requires different prompting strategies for different datasets
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--bce", action='store_true', help='whether to use BCE loss')
    parser.add_argument("--epsilon", type=int, default=16, help='lp ball limit for PGD attack')
    parser.add_argument("--alpha", type=float, default=2, help='step size for PGD attack')
    parser.add_argument("--num-iter", type=int, default=10, help='number of iterations for PGD attack')
    parser.add_argument("--api-key", type=str, default=None, help='OpenAI API key')
    
    parser.add_argument('--positive-few-shot', type=bool, default=False, help='whether to include positive examplar')
    parser.add_argument('--negative-few-shot', type=bool, default=False, help='whether to include negative examplar')
    
    args = parser.parse_args()

    main(args)
