import json
import argparse
from eval_utils import *
import tqdm.auto as tqdm

def evaluate_yes_no(test_data):

    pos = 'yes'
    neg = 'no'
    TP, TN, FP, FN = 0, 0, 0, 0

    for item in tqdm.tqdm(test_data):

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

    precision = float(TP) / float(TP + FP) if TP + FP > 0 else 0
    recall = float(TP) / float(TP + FN) if TP + FN > 0 else 0
    # f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    # print('F1 score: {}'.format(f1))


def main(args):
    with open(args.predictions_file, 'r') as file:
        test_data = json.load(file)

    evaluate_yes_no(test_data)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process answers and questions files.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions file.')
    parser.add_argument('--ans_key', type=str, required=True, help='Key of the answer you want to evaluate in the dict of the results, like llava_answer.')
    args = parser.parse_args()

    main(args)