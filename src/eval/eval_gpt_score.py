import json
import tqdm
from pprint import pprint
from collections import defaultdict
import argparse
import time
import base64
import requests
import os

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def gpt4(content: str, api_key=None, proxy='openai', image_path=None):
    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    if image_path is not None:
        base64_image = encode_image(image_path)
        params = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional driving expert."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],

            "model": 'gpt-4o-2024-08-06',
            "temperature": 0.2,  
        }
    else:
        params = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional driving expert."
                },

                {
                    "role": 'user',
                    "content": content
                }
            ],
            "model": 'gpt-4o-2024-08-06',
            "temperature": 0.2,  
        }
    received = False
    while not received:
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=params,
                stream=False
            )
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            received = True
        except:
            time.sleep(1)
    return res_content

def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


class ChatEvaluation:
    @staticmethod
    def get_avg(x):
        return sum(x) / len(x) if len(x) > 0 else 0

    @staticmethod
    def eval(samples):
        score_dict = defaultdict(list)
        for sample in samples:
            scores =  sample.get('scores', '')
            if len(scores) == 2 and all(str(s).replace('.', '', 1).isdigit() for s in scores):
                # gpt4_score = max(1, min(10, float(scores[0])))
                gpt4_score = min(1,float(scores[1]/scores[0]))
                score_dict['gpt4_score'].append(gpt4_score)
            else:
                print(f"Skipping evaluation for sample with unexpected scores: {scores}")

        result = {}
        for key, scores in score_dict.items():
            result[key] = ChatEvaluation.get_avg(scores)
        result['data_size'] = len(score_dict.get('gpt4_score', []))

        return result

def process_jsonl(args):
    samples = []
    with open(args.questions_file, 'r') as infile1:
        questions = json.load(infile1)
        total_lines = len(questions)

    with open(args.questions_file, 'r') as infile1, open(args.predictions_file, 'r') as infile2:
        cnt = 0
        questions = json.load(infile1)
        predictions = json.load(infile2)
        for data1, data2 in tqdm.tqdm(zip(questions, predictions),total=total_lines):

            if not data1["question_type"] == "open-ended":
                continue

            prompt = f"""We would like to request your feedback on the performance of two answers in response to the user question displayed above. The user asks the question on observing a driving scene.\nPlease rate the helpfulness, relevance, accuracy, and level of details of their responses. Each answer receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease output only a single line containing only two values indicating the scores for Answer 1 and 2, respectively. The two scores are separated by a space.\n
            Question: {data1['question']} \n 
            Answer 1: {data1['answer']} \n 
            Answer 2: {data2[args.ans_key]} \n
            """

            image_file = data1['image_path']
            image_path = os.path.join(args.image_folder, image_file)

            scores = gpt4(prompt, image_path=image_path)
            scores = parse_score(scores)
                
            new_sample = {
                'question': data1['question'],
                'answer': data1['answer'],
                'model_output': data2[args.ans_key],
                'scores': scores
            }
            samples.append(new_sample)
            cnt += 1

            with open(args.output_file, 'a') as outfile:
                json.dump(new_sample, outfile)
                outfile.write('\n')

        print("Num Open-ended Questions:\t", cnt)

        evaluation_results = ChatEvaluation.eval(samples)
        pprint(evaluation_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Model Response based on the provided biomedical Question and Answer.')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to the questions file.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the predictions file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the images.')
    parser.add_argument('--ans_key', type=str, required=True, help='Key of the answer you want to evaluate in the dict of the results, like llava_answer.')
    args = parser.parse_args()

    process_jsonl(args)
