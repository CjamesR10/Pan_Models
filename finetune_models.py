from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np
import glob
import evaluate
import json
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dic: dict, tokenizer: AutoTokenizer):
        self.encodings = tokenizer(
            [pair[0] + tokenizer.sep_token + pair[1] for pair in dic.keys()], 
            truncation=True, 
            padding=True, 
            return_tensors="pt",
            max_length=512
            )
        self.labels = list(dic.values())

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def read_files(path):
    out_dic = {}
    for file in glob.glob(os.path.join(path, "problem-*.txt")):
        file_id = os.path.basename(file)[8:-4]
        pairs = []
        # read input files
        with open(file, "r", newline="", encoding="utf-8") as fd:
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
        # get ground truth labels
        truth_file = "truth-problem-" + file_id + ".json"
        with open(os.path.join(path, truth_file), "r", newline="", encoding="utf-8") as fd:
            ground_truth = json.load(fd)
        changes = ground_truth["changes"]
        # append to output dictionary
        for i in range(len(pairs)):
            out_dic[pairs[i]] = changes[i]
    return out_dic

def finetune_model(data_dict: dict, tokenizer):
    # select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # load pre-trained model
    model_name = data_dict['name']
    model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-large', num_labels=2).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # create training args and trainer
    training_args = TrainingArguments(
        output_dir=model_name,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_dict['train'],
        eval_dataset=data_dict['eval'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("training", model_name)
    trainer.train()
    trainer.save_model()

def main():
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large')
    easy = {
        'name': 'roberta_task1_finetuned',
        'train': CustomDataset(read_files("./data/easy/train/"), tokenizer), 
        'eval': CustomDataset(read_files("./data/easy/validation/"), tokenizer)
    }
    finetune_model(easy, tokenizer)

    medium = {
        'name': 'roberta_task2_finetuned',
        'train': CustomDataset(read_files("./data/medium/train/"), tokenizer), 
        'eval': CustomDataset(read_files("./data/medium/validation/"), tokenizer)
    }
    finetune_model(medium, tokenizer)
    
    hard = {
        'name': 'roberta_task3_finetuned',
        'train': CustomDataset(read_files("./data/hard/train/"), tokenizer), 
        'eval': CustomDataset(read_files("./data/hard/validation/"), tokenizer)
    }
    finetune_model(hard, tokenizer)

if __name__ == "__main__":
    main()