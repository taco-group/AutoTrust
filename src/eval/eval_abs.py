import json
import argparse

def is_refused_response(response):

    if isinstance(response, list):
            response = response[0]

    refused_phrases = ["can't answer", "no answer", "cannot answer", "not appropriate",
                        "refuse to answer", "sorry", "inappropriate",
                        "apologize", "unable to", "no need", "not able", "none of", "not sure", "cannot provide", "can't provide"]
    # ["can't answer", "no answer", "cannot answer", "not appropriate",
    #                     "refuse to answer", "sorry", "inappropriate", "cannot provide",
    #                     "apologize", "not visible"]
    return any(phrase in response.lower() for phrase in refused_phrases)

def main(args):

    test_data_path = args.test_data_path

    with open(test_data_path, 'r') as file:
        test_data = json.load(file)
    
    print("Setup Data Already")

    refused_count = sum(1 for entry in test_data if is_refused_response(entry[args.ans_key]))

    total_responses = len(test_data)
    refusal_rate = refused_count / total_responses if total_responses > 0 else 0

    print(f"NUM OF REFUSAL ANSWER: {refused_count}")
    print(f"NUM OF TOTAL: {total_responses}")
    print(f"REFUSAL RATE: {refusal_rate}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Model Response based on the provided paths.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data JSONL file.')
    parser.add_argument('--ans_key', type=str, required=True, help='Kay of the answer you want to evaluate in the dict of the results, like llava_answer.')
    args = parser.parse_args()

    main(args)
