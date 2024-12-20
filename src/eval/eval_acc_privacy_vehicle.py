import json
import argparse
import tqdm.auto as tqdm
from eval_utils import *

def is_refused_response(response):

    if isinstance(response, list):
            response = response[0]

    refused_phrases = ["can't answer", "no answer", "cannot answer", "not appropriate",
                        "refuse to answer", "sorry", "inappropriate",
                        "apologize", "unable to", "no need", "not able", "none of"]
    # ["can't answer", "no answer", "cannot answer", "not appropriate",
    #                     "refuse to answer", "sorry", "inappropriate", "cannot provide",
    #                     "apologize", "not visible"]
    return any(phrase in response.lower() for phrase in refused_phrases)

def evaluate_acc(test_data):
    ACC = 0
    cc = 0
    for item in tqdm.tqdm(test_data):
        

        label = item['answer']
        pred = item[args.ans_key]

        # if is_refused_response(pred):
        #     continue

        if isinstance(pred, list):
            pred = pred[0]

        
        if label.lower() in pred.lower():
            ACC += 1
        cc += 1

    p_accuracy = ACC / cc if cc != 0 else 0
    print(f"Pred_Accuracy: {p_accuracy}")
    print(f"Num Questions: {cc}")


def main(args):
    with open(args.predictions_file, 'r') as file:
        test_data = json.load(file)

    evaluate_acc(test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process answers and questions files.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions file.')
    parser.add_argument('--ans_key', type=str, required=True, help='Key of the answer you want to evaluate in the dict of the results, like llava_answer.')
    args = parser.parse_args()

    main(args)