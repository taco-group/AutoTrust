import json
import argparse
import tqdm.auto as tqdm
from eval_utils import *

with open("./Robustness/answers/robustness/VLM_OOD_data_2024_answer_drivelm_challenge.json", 'r') as file:
    ood_data = json.load(file)

question_list = [item["image_path"] for item in ood_data if "CoVLA" in item["image_path"]]
question_list = [image_path.replace("./CoVLA/", '') for image_path in question_list]

# question_list = list(set(question_list))

def evaluate_multiple_choice(test_data, args=None):
    ACC = 0
    cc = 0
    for item in tqdm.tqdm(test_data):

        item["question"] = item["question"].replace('\n', ' ')

        image = item["image_path"]
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

        if 'CoVLA' in args.predictions_file or 'NuScenesQA' in args.predictions_file or "VLM_OOD" in args.predictions_file:
            check = True
        else:
            check = False
        if pred.lower() not in label_pool:
            Choice_list = get_options(item["question"], check=check)
            index_pred = find_most_similar_index(Choice_list, pred)
            pred = label_pool[index_pred]

        if pred.lower() == label.lower():
            ACC += 1
        cc += 1

    p_accuracy = ACC / cc if cc != 0 else 0
    return ACC, p_accuracy, cc
    # print(f"Pred_Accuracy: {p_accuracy}")
    # print(f"Num Questions: {cc}")

def evaluate_yes_no(test_data):

    pos = 'yes'
    neg = 'no'
    TP, TN, FP, FN = 0, 0, 0, 0

    for item in tqdm.tqdm(test_data):

        image = item["image_path"]
        if not image in question_list:
            continue

        if item['question_type'] == 'multi-choice':
            continue
        if item['question_type'] == 'open-ended':
            continue
        
        label_pool = ['yes', 'no']

        label = item['answer']
        pred = item[args.ans_key]

        if isinstance(pred, list):
            pred = pred[0]

        # Only keep the first sentence
        if pred.find('.') != -1:
            pred = pred.split('.')[0]

        pred = pred.replace(',', '')
        words = pred.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            pred = 'no'
        else:
            pred = 'yes'
        label = label.lower()

        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    # precision = float(TP) / float(TP + FP)
    # recall = float(TP) / float(TP + FN)
    # f1 = 2*precision*recall / (precision + recall)
    ACC = TP + TN
    cc = TP + TN + FP + FN
    p_accuracy = (TP + TN) / (TP + TN + FP + FN)
    # print('Accuracy: {}'.format(acc))
    # print('Precision: {}'.format(precision))
    # print('Recall: {}'.format(recall))
    # # print('F1 score: {}'.format(f1))
    return ACC, p_accuracy, cc


def main(args):
    with open(args.predictions_file, 'r') as file:
        test_data = json.load(file)

    multi_acc, multi_p_accuracy, multi_cc = evaluate_multiple_choice(test_data, args=args)
    yesno_acc, yesno_p_accuracy, yesno_cc = evaluate_yes_no(test_data)

    print(f"Multi_Pred_Accuracy: {multi_p_accuracy}")
    print(f"Num Multi Questions: {multi_cc}")

    print(f"Yesno_Pred_Accuracy: {yesno_p_accuracy}")
    print(f"Num Yesno Questions: {yesno_cc}")


    close_cc = multi_cc + yesno_cc
    close_acc = (multi_acc + yesno_acc) / close_cc

    print(f"Close_Pred_Accuracy: {close_acc}")
    print(f"Num Close Questions: {close_cc}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process answers and questions files.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions file.')
    parser.add_argument('--ans_key', type=str, required=True, help='Key of the answer you want to evaluate in the dict of the results, like llava_answer.')
    args = parser.parse_args()

    main(args)