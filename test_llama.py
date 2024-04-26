import transformers
import torch
import os
import json
import glob
import argparse
from tqdm import tqdm
if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

def evaluate_file(path: str, pipeline, sys_message: str) -> list:
    pairs = []
    # read input file
    with open(path, "r", newline="", encoding="utf-8") as fd:
        lines = fd.readlines()
        paragraphs = []
        # handle multiline paragraphs
        for line in lines:
            if line.startswith(" "):
                paragraphs[-1] = paragraphs[-1].join(line)
            else:
                paragraphs.append(line)
        # create paragraph pairs
        for i in range(1, len(paragraphs)):
            pairs.append((paragraphs[i - 1].strip() + '\n' + paragraphs[i].strip()))
    # evaluate paragraph pairs
    predictions = []
    for pair in pairs:
        pred = get_response(pipeline, sys_message, pair)
        match pred.strip():
            case '1':
                pred = 1
            case '0':
                pred = 0
            case _:
                print(f"Improper response {pred}")
                pred = -1
        predictions.append(pred)
    return predictions

def run_task(in_dir: str, model_id: str, sys_message: str) -> dict:
    pipeline = transformers.pipeline(
        'text-generation',
        model=model_id,
        model_kwargs={'torch_dtype': torch.bfloat16},
        device_map='auto'
    )
    solutions = {}
    for file in tqdm(glob.glob(os.path.join(in_dir, 'problem-*.txt'))):
        file_id = os.path.basename(file)[8:-4]
        predictions = evaluate_file(file, pipeline, sys_message)
        solutions[file_id] = predictions
    return solutions

def get_response(pipeline, sys_message: str, user_message: str) -> str:
    # generate prompt
    messages = [
        {'role': 'system', 'content': sys_message},
        {'role': 'user', 'content': user_message}
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # pass to pipeline
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=1,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    return outputs[0]['generated_text'][len(prompt):]

def write_results(output: dict, dir: str):
    for f_id, predictions in output.items():
        solution_name = 'solution-problem-' + f_id + '.json'
        with open(os.path.join(dir, solution_name), 'w', encoding='utf-8') as fp:
            json.dump({"changes": predictions}, fp=fp)

def read_truth_files(truth_dir: str) -> dict:
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_dir, 'truth-problem*.json')):
        with open(truth_file, 'r', encoding='utf-8') as fp:
            curr_truth = json.load(fp)
            truth[os.path.basename(truth_file)[6:-5]] = curr_truth
    return truth

def main():
    parser = argparse.ArgumentParser(description="Llama3 Style Change Detection Classifier")
    parser.add_argument("-i", "--input", 
                        help="path to the dir holding the problem files (in a dir for each dataset/task)",  
                        default='.')
    parser.add_argument('-o', '--output',
                        help="path to the dir to write solution files to", 
                        default='.')
    parser.add_argument('-e', '--evaluate',
                        help="whether to evaluate results or not. Will not write solutions to output dir",
                        action='store_true')
    parser.add_argument('-t', '--truth',
                        help='path to the dir holding the true labels (in a folder for each dataset/task)',
                        default=None)
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    truth_dir = args.truth if args.truth else in_dir

    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    sys_message = "You are an expert in writing style analysis. When you are given two paragraphs, say '0' if they are written by the same person, or '1' if they are not. Do not explain your reasoning."

    # get results
    task1 = run_task(os.path.join(in_dir, "easy/train"), model_id, sys_message)
    task2 = run_task(os.path.join(in_dir, "medium/train"), model_id, sys_message)
    task3 = run_task(os.path.join(in_dir, "hard/train"), model_id, sys_message)
    if args.evaluate:
        # evaluate
        pass
    else:
        # print
        write_results(task1, os.path.join(out_dir, 'easy'))
        write_results(task2, os.path.join(out_dir, 'medium'))
        write_results(task3, os.path.join(out_dir, 'hard'))


if __name__ == '__main__':
    main()