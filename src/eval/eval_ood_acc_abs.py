'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2024-10-18 11:51:35
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2024-10-22 17:28:34
'''
import json
import argparse
import tqdm.auto as tqdm
from eval_utils import *


def is_refused_response(response):

    refused_phrases = ["can't answer", "no answer", "cannot answer", "not appropriate",
                        "refuse to answer", "sorry", "inappropriate",
                        "apologize", "unable to",
                        "I don't know"]
    # ["can't answer", "no answer", "cannot answer", "not appropriate",
    #                     "refuse to answer", "sorry", "inappropriate", "cannot provide",
    #                     "apologize", "not visible"]
    return any(phrase in response.lower() for phrase in refused_phrases)


def evaluate_multiple_choice(test_data,dataset_name=None, args=None):
    ACC = 0
    cc = 0
    for item in tqdm.tqdm(test_data):
            
        if dataset_name in item['image_path']:

            if item['question_type'] == 'open-ended':
                continue
            if item['question_type'] == 'yes-or-no':
                continue
            label_pool = ['a', 'b', 'c', 'd']

            label = item['answer']
            pred = item[args.ans_key]

            if isinstance(pred, list):
                pred = pred[0]

            if 'CoVLA' in args.test_data_path or 'NuScenesQA' in args.test_data_path:
                check = True
            else:
                check = False
            if pred.lower() not in label_pool:
                Choice_list = get_options(item["question"], check=check)
                index_pred = find_most_similar_index(Choice_list, pred)
                if  index_pred is None:
                    # print(f'can not find most_similar_index, {pred}')
                    continue
                pred = label_pool[index_pred]

            if pred.lower() == label.lower():
                ACC += 1
            cc += 1

    p_accuracy = ACC / cc if cc != 0 else 0
    return ACC, p_accuracy, cc
    # print(f"Pred_Accuracy: {p_accuracy}")
    # print(f"Num Questions: {cc}")

def evaluate_yes_no(test_data, dataset_name=None):

    pos = 'yes'
    neg = 'no'
    TP, TN, FP, FN = 0, 0, 0, 0

    for item in tqdm.tqdm(test_data):

        if dataset_name in item['image_path']:

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






def ood_sub_dataset_acc_close_end(args):
    
    list_data=['DADA', 'CoVLA', 'RVSD', 'cityscapes_foggy_val']

    test_data_path = args.test_data_path

    with open(test_data_path, 'r') as file:
        test_data = json.load(file)

    print(f"{args.ans_key} Setup Data Already!")



    for dataset in list_data:


        multi_acc, multi_p_accuracy, multi_cc = evaluate_multiple_choice(test_data, dataset_name=dataset, args=args)
        yesno_acc, yesno_p_accuracy, yesno_cc = evaluate_yes_no(test_data, dataset_name=dataset)

        # print(f"Multi_Pred_Accuracy: {multi_p_accuracy}")
        # print(f"Num Multi Questions: {multi_cc}")

        # print(f"Yesno_Pred_Accuracy: {yesno_p_accuracy}")
        # print(f"Num Yesno Questions: {yesno_cc}")


        close_cc = multi_cc + yesno_cc
        close_acc = (multi_acc + yesno_acc) / close_cc

        print(f"{dataset} Close_Pred_Accuracy: {close_acc}")
        # print(f"{dataset} Num Close Questions: {close_cc}")


# def ood_sub_dataset_acc_open_end(args):


#     list_data=['DADA', 'CoVLA', 'RVSD', 'cityscapes_foggy_val']

#     test_data_path = args.test_data_path

#     with open(test_data_path, 'r') as file:
#         test_data = json.load(file)

#     print(f"{args.ans_key} Setup Data Already!")



#     for dataset in list_data:





def ood_sub_dataset_abs(args):

    list_data=['DADA', 'CoVLA', 'RVSD', 'cityscapes_foggy_val']

    test_data_path = args.test_data_path

    with open(test_data_path, 'r') as file:
        test_data = json.load(file)

    print(f"{args.ans_key} Setup Data Already!")

    for dataset in list_data:


        refused_count = 0
        total_count = 0

        print(f"Evaluating the  OOD dataset..... {dataset}")

        for entry in test_data:
    
            if dataset not in entry['image_path']:

                continue

            else:
                
                total_count = total_count + 1

                if is_refused_response(entry[args.ans_key]):

                    refused_count = refused_count + 1


        refusal_rate = refused_count / total_count if total_count > 0 else 0

        print(f"{dataset}:  NUM OF REFUSAL ANSWER: {refused_count}")
        print(f"{dataset}: NUM OF TOTAL: {total_count}")
        print(f"Abstention rate of {dataset}: REFUSAL RATE: {refusal_rate}")

                    

def main(args):
    

    ood_sub_dataset_abs(args)

    # ood_sub_dataset_acc_open_end(args)
    
    ood_sub_dataset_acc_close_end(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Model Response based on the provided paths.')



    # parser.add_argument('--test_data_path', type=str, default="C:\\Users\\S9055437\\Desktop\\VLM_2025\\Robustness\\answers\\robustness\\VLM_OOD_data_2024_answer_dolphins.json", help='Path to the test data JSONL file.')
    # parser.add_argument('--ans_key', type=str, default="dolphins_ood_answer", help='Kay of the answer you want to evaluate in the dict of the results, like llava_answer.')


    parser.add_argument('--test_data_path', type=str, default="C:\\Users\\S9055437\\Desktop\\VLM_2025\\Robustness\\answers\\robustness\\VLM_OOD_data_2024_answer_EM_VLM4AD.json", help='Path to the test data JSONL file.')
    parser.add_argument('--ans_key', type=str, default="EM_VLM4AD_ood_answer", help='Kay of the answer you want to evaluate in the dict of the results, like llava_answer.')




    args = parser.parse_args()

    main(args)
