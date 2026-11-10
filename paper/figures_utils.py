import pandas as pd
import numpy as np
import json
import os

def read_sweep_data(base_dir, max_runs, variable_key, data_key):
    """
    Reads the result of a sweep with up to max_runs repeats.

    :param base_dir: A directory where the results are stored. Contains folders run_i, with contents results_run_j.csv
                     and params_run_j.csv. This naming may change in the future to be clearer.
    :param max_runs: Reads up to max_runs repeats.
    :param variable_key: A string containing the variable name that is being swept.
    :param data_key: Reads the value for the data key.
    :return variable_values, data_obj: variable_values is a sorted list of all variable values seen
                                       data_obj is an object like {variable_value: [data_key_values, ...]}
    """
    # Data containers
    variable_values_set = set()
    data = {}  # var_val -> list of data

    # Load results by repeat
    for repeat_i in range(max_runs):
        # Form the directory for the current repeat and skip if it doesn't exist
        repeat_dir = os.path.join(base_dir, f'run_{repeat_i}')

        if not os.path.isdir(repeat_dir):
            print(f'Skipping repeat {repeat_i} for base directory {base_dir}')
            continue

        # Find all results CSVs in this run folder
        results_csvs = [f for f in os.listdir(repeat_dir) if f.startswith('results_run_') and f.endswith('.csv')]

        for csv_file in results_csvs:
            # Get the results path and corresponding parameters path
            results_path = os.path.join(repeat_dir, csv_file)
            params_file = os.path.join(repeat_dir, csv_file.replace('results_', 'params_').replace('.csv', '.json'))
            if not os.path.exists(params_file):
                print(f'Did not find {params_file} for {csv_file}, skipping')
                continue

            # Load sweep variable
            params_json = json.load(open(params_file, 'r'))
            var_val = params_json[variable_key]
            variable_values_set.add(var_val)

            # Initialize containers
            if var_val not in data:
                data[var_val] = []

            # Load results
            results_df = pd.read_csv(results_path)
            data[var_val].append(results_df[data_key].mean())

    # Sort variable values
    variable_values = sorted(variable_values_set)

    return variable_values, data

def compute_means_stds(variable_values, data):
    """
    Computes the means and standard deviations of all variable values.

    :param variable_values: A sorted list of all variable values which data is indexed by.
    :param data: A dict like {variable_value: [data_key_values, ...]}
    :return means, stds: A list of means and stds corresponding to the order of variable values
    """
    means = []
    stds = []

    for var_val in variable_values:
        data_repeats = data.get(var_val, [])

        if data_repeats is not None:
            means.append(np.mean(data_repeats))
            stds.append(np.std(data_repeats))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    return means, stds