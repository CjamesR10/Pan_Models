import os
import random
from sklearn.metrics import f1_score
from itertools import chain
from tqdm import tqdm
import glob
import json

def bootstrap_test(test_data, base_data, truth_data, num_samples):
    
    sigma = (f1_score(list(chain.from_iterable(truth_data)), list(chain.from_iterable(test_data)), average='macro', labels=[0,1], zero_division=0) -
            f1_score(list(chain.from_iterable(truth_data)), list(chain.from_iterable(base_data)), average='macro', labels=[0,1], zero_division=0))

    s = 0
    for _ in tqdm(range(num_samples)):
        test_sample = []
        base_sample = []
        truth_sample = []
        for _ in range(len(truth_data)):
            rand = random.randint(0, len(truth_data) - 1)
            truth_sample.append(truth_data[rand])
            test_sample.append(test_data[rand])
            base_sample.append(base_data[rand])
        if sigma > 0:
            if (f1_score(list(chain.from_iterable(truth_sample)), list(chain.from_iterable(test_sample)), average='macro', labels=[0,1], zero_division=0) -
                f1_score(list(chain.from_iterable(truth_sample)), list(chain.from_iterable(base_sample)), average='macro', labels=[0,1], zero_division=0)) >= (2 * sigma):
                s += 1
        elif (f1_score(list(chain.from_iterable(truth_sample)), list(chain.from_iterable(base_sample)), average='macro', labels=[0,1], zero_division=0) -
              f1_score(list(chain.from_iterable(truth_sample)), list(chain.from_iterable(test_sample)), average='macro', labels=[0,1], zero_division=0)) >= (-2 * sigma):
                s += 1
        
    return s / num_samples


def read_solution_files(solutions_folder: str) -> dict:
    solutions = {}
    for solution_file in glob.glob(os.path.join(solutions_folder, 'solution-problem-*.json')):
        with open(solution_file, 'r') as fh:
            curr_solution = json.load(fh)
            solutions[os.path.basename(solution_file)[9:-5]] = curr_solution
    return solutions


def read_ground_truth_files(truth_folder: str) -> dict:
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-problem*.json')):
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[6:-5]] = curr_truth
    return truth


def extract_task_results(truth: dict, solutions: dict, base: dict) -> tuple:
    all_solutions = []
    all_truth = []
    all_base = []
    for problem_id, truth_instance in sorted(truth.items()):
        if problem_id not in solutions.keys() or problem_id not in base.keys():
            continue
        if len(truth_instance['changes']) != len(solutions[problem_id]['changes']) or len(truth_instance['changes']) != len(base[problem_id]['changes']):
            continue
        all_truth.append(truth_instance['changes'])
        all_solutions.append(solutions[problem_id]['changes'])
        all_base.append(base[problem_id]['changes'])
    return all_truth, all_solutions, all_base


def main():
    truth_dir = 'data'
    llama_dir = 'llama_solutions'
    orca_dir = 'orca_solutions'
    base_dir = 'base_solutions'

    tasks = ['easy', 'medium', 'hard']

    for task in tasks:
        truth = read_ground_truth_files(os.path.join(truth_dir, task, 'validation'))
        base = read_solution_files(os.path.join(base_dir, task))
        llama = read_solution_files(os.path.join(llama_dir, task))
        orca = read_solution_files(os.path.join(orca_dir, task))
        
        truth_temp, orca_temp, base_temp = extract_task_results(truth, orca, base)
        score = bootstrap_test(base_temp, orca_temp, truth_temp, 10000)
        print(f"Orca vs Baseline for {task}: {score}")
        truth_temp, llama_temp, base_temp = extract_task_results(truth, llama, base)
        score = bootstrap_test(base_temp, llama_temp, truth_temp, 10000)
        print(f"Llama vs Baseline for {task}: {score}")
        truth_temp, llama_temp, orca_temp = extract_task_results(truth, llama, orca)
        score = bootstrap_test(llama_temp, orca_temp, truth_temp, 10000)
        print(f"Llama vs Orca for {task}: {score}")


if __name__ == '__main__':
    main()