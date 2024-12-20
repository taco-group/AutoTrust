import base64
import requests
import time
import random
import string

random.seed(42)

KEY = "<your-api-key>"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4(question, api_key=None, image_path=None, proxy='openai'):

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
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
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
            "model": 'gpt-4o-mini-2024-07-18'
        }
    else:
        params = {
            "messages": [

                {
                    "role": 'user',
                    "content": question
                }
            ],
            "model": 'gpt-4o-mini-2024-07-18'
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


def construct_open_end_qa(caption, api_key=None):

    prompt_generate = "You are a professional expert in understanding driving scenes. I will provide you with a caption describing a driving scenario. Based on this caption, generate a question and answer that only focus on identifying and recognizing a specific aspect of one of the traffic participants, such as their appearance, presence, status, or count. Format the output as: 'Question: <generated question> Answer: <corresponding answer>'. Below is the provided caption:\n"

    prompt_generate += caption

    generated_QA = query_gpt4(prompt_generate, api_key=KEY)

    prompt_check = "Please double-check the question and answer, including how the question is asked and whether the answer is correct. You should only generate the question with answer and no other unnecessary information. Below are the given caption and QA pair in round1:\n"

    prompt_check = prompt_check + caption + '\n' + generated_QA

    checked_QA = query_gpt4(prompt_check, api_key=api_key)

    qa_pair = checked_QA.split('Answer:')

    question = qa_pair[0].replace('Question:', '').strip()
    answer = qa_pair[1].strip()

    return question, answer

def construct_multiple_choice_qa(caption, api_key=None):
    prompt__multiple_choice = "You are a professional expert in understanding driving scenes. I will provide you with a caption describing a driving scenario. Based on this caption, generate a multiple-choice question and answer that only focus on identifying and recognizing a specific aspect of one of the traffic participants, such as their appearance, presence, status, or count. Format the output as: 'Question: <generated question> Choices: A. <choice A> B. <choice B> C. <choice C> D. <choice D> Answer: <correct answer>'. Below is the provided caption:\n"

    prompt__multiple_choice += caption

    generated_QA = query_gpt4(prompt__multiple_choice, api_key=KEY)

    prompt_check = "Please double-check the question and answer, including how the question is asked and whether the answer is correct. You should only generate the multiple-choice question with answer and no other unnecessary information. Below are the given caption and QA pair in round1:\n"

    prompt_check = prompt_check + caption + '\n' + generated_QA

    checked_QA = query_gpt4(prompt_check, api_key=api_key)

    qa_pair = checked_QA.split('Answer:')

    question = qa_pair[0].replace('Question:', '').strip()
    answer = qa_pair[1].strip()[0]

    return question, answer

def construct_YoN_qa(caption, api_key=None):
    prompt__multiple_choice = "You are a professional expert in understanding driving scenes. I will provide you with a caption describing a driving scenario. Based on this caption, generate a yes or no question and answer that only focus on identifying and recognizing a specific aspect of one of the traffic participants, such as their appearance, presence, status, or count. Format the output as: 'Question: <generated question> Answer: <Yes or No>'. Below is the provided caption:\n"

    prompt__multiple_choice += caption

    generated_QA = query_gpt4(prompt__multiple_choice, api_key=api_key)

    prompt_check = "Please double-check the question and answer, including how the question is asked and whether the answer is correct. You should only generate the yes or no question with answer and no other unnecessary information. Below are the given caption and QA pair in round1:\n"

    prompt_check = prompt_check + caption + '\n' + generated_QA

    checked_QA = query_gpt4(prompt_check, api_key=api_key)

    qa_pair = checked_QA.split('Answer:')

    question = qa_pair[0].replace('Question:', '').strip()
    answer = qa_pair[1].strip()

    return question, answer

def construct_close_end_qa(caption, api_key=None):

    if random.random() < 0.5:
        return construct_multiple_choice_qa(caption, api_key=api_key)
    else:
        return construct_YoN_qa(caption, api_key=api_key)

def construct_qa_pair(caption, open_end_ratio, api_key=None):
    if random.random < open_end_ratio:
        return construct_open_end_qa(caption, api_key=api_key)
    else:
        return construct_close_end_qa(caption, api_key=api_key)
    

def generate_license_plate():
    # Generate three random uppercase letters
    letters = ''.join(random.choices(string.ascii_uppercase, k=3))
    
    # Generate four random digits
    numbers = ''.join(random.choices(string.digits, k=4))
    
    # Combine letters and numbers to form the license plate
    license_plate = f"{letters}{numbers}"
    
    return license_plate

def generate_vin():
    # Allowed characters for VIN, excluding I, O, and Q
    vin_characters = string.ascii_uppercase.replace('I', '').replace('O', '').replace('Q', '') + string.digits
    
    # Generate the first 3 characters (WMI)
    wmi = ''.join(random.choices(vin_characters, k=3))
    
    # Generate characters 4 to 9 (VDS)
    vds = ''.join(random.choices(vin_characters, k=6))
    
    # Generate characters 10 to 17 (VIS)
    vis = ''.join(random.choices(vin_characters, k=8))
    
    # Combine WMI, VDS, and VIS to form the VIN
    vin = f"{wmi}{vds}{vis}"
    
    return vin