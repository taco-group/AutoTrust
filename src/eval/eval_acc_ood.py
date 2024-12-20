import json
import argparse
import tqdm.auto as tqdm
from eval_utils import *


with open("./Robustness/answers/robustness/VLM_OOD_data_2024_answer_drivelm_challenge.json", 'r') as file:
    ood_data = json.load(file)

question_list = [item["image_path"] for item in ood_data if "CoVLA" in item["image_path"]]
question_list = [image_path.split('/')[-2] for image_path in question_list]

question_list = list(set(question_list))

def evaluate_multiple_choice(test_data):
    ACC = 0
    cc = 0
    for item in tqdm.tqdm(test_data):

        image = item["image_path"].split('/')[-2]
        if not image in question_list:
            continue

        if item['question_type'] == 'open-ended':
            continue
        if item['question_type'] == 'yes-or-no':
            continue
        label_pool = ['a', 'b', 'c', 'd']

        label = item['answer']
        pred = item[args.ans_key]

        if isinstance(pred, list):
            pred = pred[0]
        if pred.lower() not in label_pool:
            Choice_list = get_options(item["question"])
            index_pred = find_most_similar_index(Choice_list, pred)
            pred = label_pool[index_pred]

        if pred.lower() == label.lower():
            ACC += 1
        cc += 1

    p_accuracy = ACC / cc if cc != 0 else 0
    print(f"Pred_Accuracy: {p_accuracy}")
    print(f"Num Questions: {cc}")


def main(args):
    with open(args.predictions_file, 'r') as file:
        test_data = json.load(file)

    evaluate_multiple_choice(test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process answers and questions files.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions file.')
    parser.add_argument('--ans_key', type=str, required=True, help='Key of the answer you want to evaluate in the dict of the results, like llava_answer.')
    args = parser.parse_args()

    main(args)