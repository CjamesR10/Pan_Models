from transformers import AutoTokenizer, pipeline
import torch
import glob
from tqdm import tqdm
import json
import os
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large')
label_dic = {'LABEL_0': 0, 'LABEL_1': 1}

def evaluate_file(path, classifier):
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
            pairs.append((paragraphs[i - 1].strip(), paragraphs[i].strip()))
    # evaluate paragraph pairs
    pairs = [pair[0] + tokenizer.sep_token + pair[1] for pair in pairs]
    predictions = classifier(pairs)
    predictions = [label_dic[pred['label']] for pred in predictions]
    return predictions

def run_task(in_dir, out_dir, model_name):
    classifier = pipeline(
        'text-classification', 
        model=model_name, 
        tokenizer=tokenizer, 
        device=0, 
        max_length=512, 
        truncation=True, 
        padding=True
        )
    for file in tqdm(glob.glob(os.path.join(in_dir, 'problem-*.txt'))):
        file_id = os.path.basename(file)[8:-4]
        predictions = evaluate_file(file, classifier)
        solution_name = "solution-problem-" + file_id + ".json"
        with open(os.path.join(out_dir, solution_name), 'w', encoding='utf-8') as fp:
            json.dump({"changes": predictions}, fp=fp)

def main():
    # subtask 1
    if not os.path.exists('./solutions/easy'):
        os.makedirs('./solutions/easy')
    print('generating solutions for subtask 1')
    run_task('./data/easy/validation', './solutions/easy', 'roberta_task1_finetuned')
    # subtask 2
    if not os.path.exists('./solutions/medium'):
        os.makedirs('./solutions/medium')
    print('generating solutions for subtask 2')
    run_task('./data/medium/validation', './solutions/medium', 'roberta_task2_finetuned')
    # subtask 3
    if not os.path.exists('./solutions/hard'):
        os.makedirs('./solutions/hard')
    print('generating solutions for subtask 3')
    run_task('./data/hard/validation', './solutions/hard', 'roberta_task3_finetuned')

if __name__ == "__main__":
    main()