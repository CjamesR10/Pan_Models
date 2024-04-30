# PAN 2024 Task 1  Models

This repo contains the models for the multi-author writing style analysis task at PAN 2024. The dataset and evaluation script, data.zip and evaluator.py respectively, are available on the task's homepage [here](https://pan.webis.de/clef24/pan24-web/style-change-detection.html).

## Contents

llama_test.py and orca_test.py are used to run the Llama-3 and Orca-2 models, respectively. plot_results.py was used to generate figures for the paper. test_significance.py was used to perform paired-bootstrap significance testing on the model results.
evaluator.py is used to calculate F1 scores for the models.

## Usage

### Running the models

In order to download and run inference with the Llama 3 model, you have to have access to the Llama 3 family of models (see [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)) and you have to have your access token stored on your machine (see
[here](https://huggingface.co/docs/hub/en/security-tokens)). Once that is done, the usage for llama_test.py and orca_test.py is exactly the same.

First, extract data.zip. Next, run the command

    py llama_test.py [-e] [-i data_path] [-o output_path] [-t truth_path]

All arguments are optional. Use -i to specify the input directory. If omitted, the input directory is assumed to be ./data. 

Use -o to specify the output directory. If omitted, the output directory is assumed to be ./solutions.

Use -t to specify the ground truth directory. If omitted, the truth directory is assumed to be the same as the input directory.

Use -e to evaluate instead of print results. In this mode, the F1 scores for each subtask are printed to stdout, and the solution files are not written to the output directory.

### Evaluating the results

The script for evaluating the results, "evaluator.py" is a modified version of the evaluation script provided by the PAN 2024 team [here](https://github.com/pan-webis-de/pan-code/tree/master/clef24/multi-author-analysis/evaluator). To run the evaluator, place the "evaluator.py" script in the directory that contains the dataset and solutions subdirectories and run the command

    py evaluator.py -p ./solutions -t ./data -o .

This will create the file "evaluation.txt", which has the F1 scores of the model for the 3 subtasks.