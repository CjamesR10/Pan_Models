from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import glob
import argparse
from itertools import chain
from sklearn.metrics import f1_score
from tqdm import tqdm
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print('cuda')
else:
    torch.set_default_device('cpu')
    print('cpu')

def evaluate_file(path: str, tokenizer, model, sys_message: str) -> list:
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
        pred = get_response(tokenizer, model, sys_message, pair)
        match pred.strip():
            case '1':
                pred = 1
            case '0':
                pred = 0
            case _:
                print(f"\nImproper response {pred}")
                pred = -1
        predictions.append(pred)
    return predictions

def run_task(in_dir: str, model_id: str, sys_message: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    solutions = {}
    i = 0
    err = False
    for file in tqdm(glob.glob(os.path.join(in_dir, 'problem-*.txt'))):
        if i >= 50:
            model = None
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            i = 0
        i += 1
        # reset model context to reduce memory overhead
        file_id = os.path.basename(file)[8:-4]
        try:
            predictions = evaluate_file(file, tokenizer, model, sys_message)
        except torch.cuda.OutOfMemoryError:
            # don't just bail if we run out of memory
            # instead clear context and skip
            # this is usually tripped because of a malformed query (non-utf 8 tokens)
            err = True
        if err:
            model = None
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            predictions = None
            err = False
        if predictions: solutions[file_id] = predictions
    model = None
    torch.cuda.empty_cache()
    return solutions

def get_response(tokenizer, model, sys_message: str, user_message: str) -> str:
    # generate prompt
    messages = [
        {'role': 'system', 'content': sys_message},
        {'role': 'user', 'content': user_message}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model.device)
    # pass to pipeline
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

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
            truth[os.path.basename(truth_file)[14:-5]] = curr_truth['changes']
    return truth

def extract_task_results(truth: dict, solutions: dict) -> tuple:
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in sorted(truth.items()):
        if problem_id not in solutions.keys():
            continue
        if len(truth_instance) != len(solutions[problem_id]):
            continue
        all_truth.append(truth_instance)
        all_solutions.append(solutions[problem_id])
    return all_truth, all_solutions

def compute_score(truth_dic, pred_dic):
    truth, solutions = extract_task_results(truth_dic, pred_dic)
    truth = list(chain.from_iterable(truth))
    solutions = list(chain.from_iterable(solutions))

    return f1_score(truth, solutions, average='macro', labels=[0,1], zero_division=0)

def main():
    parser = argparse.ArgumentParser(description="Llama3 Style Change Detection Classifier")
    parser.add_argument("-i", "--input", 
                        help="path to the dir holding the problem files (in a dir for each dataset/task)",  
                        default='data')
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
    task1 = run_task(os.path.join(in_dir, "easy/validation"), model_id, sys_message)
    task2 = run_task(os.path.join(in_dir, "medium/validation"), model_id, sys_message)
    task3 = run_task(os.path.join(in_dir, "hard/validation"), model_id, sys_message)
    if args.evaluate:
        # evaluate
        task1_truth = read_truth_files(os.path.join(truth_dir, 'easy/validation'))
        task1_score = compute_score(task1_truth, task1)
        task2_truth = read_truth_files(os.path.join(truth_dir, 'medium/validation'))
        task2_score = compute_score(task2_truth, task2)
        task3_truth = read_truth_files(os.path.join(truth_dir, 'hard/validation'))
        task3_score = compute_score(task3_truth, task3)
        print(f"Task 1 score: {task1_score:.3f}")
        print(f"Task 2 score: {task2_score:.3f}")
        print(f"Task 3 score: {task3_score:.3f}")
    else:
        # print
        if not os.path.exists(os.path.join(out_dir, 'easy')):
            os.makedirs(os.path.join(out_dir, 'easy'))
        write_results(task1, os.path.join(out_dir, 'easy'))
        if not os.path.exists(os.path.join(out_dir, 'medium')):
            os.makedirs(os.path.join(out_dir, 'medium'))
        write_results(task2, os.path.join(out_dir, 'medium'))
        if not os.path.exists(os.path.join(out_dir, 'hard')):
            os.makedirs(os.path.join(out_dir, 'hard'))
        write_results(task3, os.path.join(out_dir, 'hard'))


if __name__ == '__main__':
    main()