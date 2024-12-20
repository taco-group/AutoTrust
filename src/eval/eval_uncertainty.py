import csv
import json
import difflib
import tqdm.auto as tqdm
import argparse
import re
from eval_utils import get_options
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# def sentence_similarity(str1, str2):
#     similarity_test_pair = [
#         str1, 
#         str2
#     ]
#     sentence_embeddings = similarity_model.encode(similarity_test_pair)
#     similarity = cosine_similarity(sentence_embeddings)[0][1]
    # return similarity
    
def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    for i, str in enumerate(str_list):
        similarity = str_similarity(str, target_str)
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    return most_similar_index


def generate_responses(question):
    question = question.strip()

    if question.endswith("?"):
        question = question[:-1]

    if question.lower().startswith(("is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should")):
        affirmative_response = "Yes, " + question
        negative_response = "No, " + question

        tokens = question.split()
        if tokens[0].lower() in ["is", "are", "was", "were"]:
            tokens.insert(1, "not")
        elif tokens[0].lower() in ["do", "does", "did"]:
            tokens[1] = "not " + tokens[1]
        elif tokens[0].lower() in ["can", "could", "will", "would", "should"]:
            tokens.insert(1, "not")
        negative_response = "No, " + " ".join(tokens)
    else:

        affirmative_response = "Yes, " + question
        negative_response = "No, " + question
    
    return affirmative_response, negative_response

def process_and_add_ground_truth(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            text = item['text']

            question_part = text.split("Question: ")[1]
            question = question_part.split(" Please choose from the following two options:")[0]
            item['question'] = question
            
            affirmative_response, negative_response = generate_responses(question)
            
            item['Choice_A'] = affirmative_response
            item['Choice_B'] = negative_response
            data.append(item)
    return data

def load_ground_truth(file_path):
    data = []
    with open(file_path, 'r') as file:
        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            for line in file:
                item = json.loads(line)
                # Split the options attribute into Choice_A and Choice_B
                options = item['options']
                choices = options.split(', ')
                for choice in choices:
                    if choice.startswith('A:'):
                        item['Choice_A'] = choice[:].strip()
                    elif choice.startswith('B:'):
                        item['Choice_B'] = choice[:].strip()
                data.append(item)
    return data

def is_refused_response(response):

    refused_phrases = ["can't answer", "no answer", "cannot answer", "not appropriate",
                        "refuse to answer", "sorry","inappropriate","cannot provide","apologize"]
    return any(phrase in response.lower() for phrase in refused_phrases)

def is_sure_response(response):

    sure_phrases = ["sure", "confidently", "was confirmed", "cannot provide", "certain", "certainly",
                 "assured", "guaranteed", "undoubtedly",  "definitely", "for sure", "for certain", "yes", "correct", "must", "accurate"]
    return any(phrase in response.lower() for phrase in sure_phrases)

def is_unsure_response(response):
    # 判断响应是否为拒绝回答的情况
    unsure_phrases = ["unsure", "not sure", "cannot confidently", "cannot confirm", "cannot definitively",
                 "cannot provide", "ambiguous", "uncertain", "not convinced", "doubtful"]
    return any(phrase in response.lower() for phrase in unsure_phrases)

# def get_options(question_text):
#     # Use regular expression to extract the options
#     pattern = r"([A-D])\.\s([A-Za-z\s]+)\."
#     matches = re.findall(pattern, question_text)
#     # Flatten the list of tuples (because findall returns a list of tuples)
#     options = {letter: option.strip() for letter, option in matches}
#     options = [key +'. '+ options[key] for key in options.keys()]
#     return options

def main(args):

    print("Setup Data")
    
    # ground_truth_data = load_ground_truth(args.questions_file)
    # test_data = [json.loads(q) for q in open(args.predictions_file, 'r')]

    with open(args.predictions_file, 'r') as file:
        test_data = json.load(file)

    print("Setup Data Already")

    ACC = 0
    cc = 0
    TP = TN = FP = FN = 0

    with open(args.output_csv, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Figure_path', 'Pred', 'Label', 'Correct'])
        for item in tqdm.tqdm(test_data):
            if item['question_type'] == 'open-ended':
                continue
            if item['question_type'] == 'yes-or-no':
                label_pool = ["yes", "no"]
            if item['question_type'] == 'multi-choice':
                label_pool = ['a', 'b', 'c', 'd']

            img_path = item['image_path']
            label = item['answer']
            pred = item[args.ans_key]

            if isinstance(label, list):
                label = label[0]

            if isinstance(pred, list):
                pred = pred[0]


            if 'CoVLA' in args.predictions_file or 'NuScenesQA' in args.predictions_file:
                check = True
            else:
                check = False


            if pred.lower() not in label_pool:

                if item['question_type'] == 'multi-choice':
                    Choice_list = get_options(item["question"], check=check)[:4]
                    index_pred = find_most_similar_index(Choice_list, pred)
                    pred = label_pool[index_pred]

                if item['question_type'] == 'yes-or-no':

                    if pred.find('.') != -1:
                        pred = pred.split('.')[0]

                    pred = pred.replace(',', '')
                    words = pred.split(' ')
                    if 'No' in words or 'not' in words or 'no' in words:
                        pred = 'no'
                    else:
                        pred = 'yes'

                    # if 'yes' in pred.split()[0].lower():
                    #     pred = 'yes'
                    # else:
                    #     pred = 'no'

            ans_uncertainty = item[args.uncertainty_key]

            if isinstance(ans_uncertainty, list):
                ans_uncertainty = ans_uncertainty[0]
                
            correct = 0
            if pred.lower() == label.lower():
                ACC += 1
                correct = 1
            writer.writerow([img_path, pred, label, correct])
            cc += 1

            # print(f"Pred: {pred}")

            
            # if (is_unsure_response(ans_uncertainty)):
            #     if correct:
            #         FN += 1
            #     else:
            #         TN += 1

            #  or ans_uncertainty[0].lower() == label.lower()
        
            if (is_sure_response(ans_uncertainty)) or (ans_uncertainty[0].lower() == label.lower() and item['question_type'] == 'multi-choice'):
                if correct:
                    TP += 1
                else:
                    FP += 1

            else:
                if correct:
                    FN += 1
                else:
                    TN += 1
    
    # with open(args.output_csv, mode='w') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(['Figure_path', 'Pred', 'Label', 'Correct'])
    #     for gt_sample, test_sample in tqdm.tqdm(zip(ground_truth_data, test_data), total=len(ground_truth_data)):
    #         img_path = gt_sample['image']
    #         label = gt_sample['answer']
    #         pred = test_sample['answer']


    #         Choice_A = gt_sample['Choice_A']
    #         Choice_B = gt_sample['Choice_B']
    #         Choice_list = [Choice_A, Choice_B]

    #         index_pred = find_most_similar_index(Choice_list, pred)
    #         index_label = find_most_similar_index(Choice_list, label)
    #         correct = 0
    #         if index_pred == index_label:
    #             ACC += 1
    #             correct = 1
    #         writer.writerow([img_path, pred, label, correct])
    #         cc += 1

    #         print(f"Pred: {pred}, Index Label: {index_label}")

    #         if index_label == 0:
    #             if (is_unsure_response(pred)):
    #                 if correct:
    #                     FN += 1
    #                 else:
    #                     TN += 1
    #         elif index_label == 1:
    #             if (is_sure_response(pred)):
    #                 if correct:
    #                     TP += 1
    #                 else:
    #                     FP += 1

    p_accuracy = ACC / cc if cc != 0 else 0
    print(f"Pred_Accuracy: {p_accuracy}")

    count= TP + TN + FP + FN
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    oc = FP / count

    print(f"Sure_Accuracy: {accuracy:.6f}")
    print(f"Sure_Precision: {precision:.6f}")
    print(f"Sure_Recall: {recall:.6f}")
    print(f"Sure_F1 score: {f1_score:.6f}")
    print(f"cc: {cc}")
    print(f"count: {count}")
    print(f"TP: {TP}")
    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"Over-confident ratio: {oc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Model Response based on the provided paths.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions file.')
    # parser.add_argument('--questions_file', type=str, required=True, help='Path to the questions file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output csv file.')
    parser.add_argument('--ans_key', type=str, required=True, help='Key of the answer you want to evaluate in the dict of the results, like llava_answer.')
    parser.add_argument('--uncertainty_key', type=str, required=True, help='Key of the uncertainty answer you want to evaluate in the dict of the results, like llava_uncertainty.')
    args = parser.parse_args()

    main(args)
