import typing
from functools import partial
import pickle
import multiprocessing
import numpy as np

import library

def plot_overlap(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, zoom: bool = True,
                 note: typing.AnyStr = None) -> None:
    library.plot_general(results, baseline=baseline, centrality=centrality, header="", zoom=zoom,
                         parabola=True, rescale=True, note=note)
    return


def get_ranking(centrality: typing.AnyStr):
    return library.get_ranking(centrality)


def simulate_one_scenario(in_centrality1: typing.AnyStr, in_centrality2: typing.AnyStr, in_configuration: typing.NamedTuple, run_index: int = None, cached_files: typing.List[str] = None):
    # If using cache, we use the .npy files corresponding to the graph
    if in_configuration.cache and cached_files is not None and run_index is not None and run_index < len(cached_files):
        file_path = cached_files[run_index]
        try:
            import os
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            npy1 = os.path.join(in_configuration.cache_dir, f"{base_name}_centrality_{in_centrality1}.npy")
            npy2 = os.path.join(in_configuration.cache_dir, f"{base_name}_centrality_{in_centrality2}.npy")

            c1_scores = np.load(npy1)
            c2_scores = np.load(npy2)

            return library.compare_centrality(c1_scores, c2_scores)

        except Exception as e:
            print(f"Error loading cached centralities for {file_path}: {e}. Falling back to generation.")
            pass

    graph = in_configuration.generator()
    return library.compare_centrality(get_ranking(in_centrality1)(graph), get_ranking(in_centrality2)(graph))


def run_comparing_sim(runs: int, centrality1: typing.AnyStr, centrality2: typing.AnyStr, configuration: typing.NamedTuple,
                      note: typing.AnyStr = None, parallel: bool = True, cached_files: typing.List[str] = None) -> None:

    print(f'Running: {centrality1} vs. {centrality2}, conffile: {configuration.name}')

    tasks = []
    for i in range(runs):
        tasks.append(partial(simulate_one_scenario, centrality1, centrality2, configuration, run_index=i, cached_files=cached_files))

    if not parallel:
        sim_results = [task() for task in tasks]
    else:
        with multiprocessing.Pool(processes=configuration.max_processes) as pool:
            worker = partial(simulate_one_scenario, centrality1, centrality2, configuration, cached_files=cached_files)
            pass

            args = [(centrality1, centrality2, configuration, i, cached_files) for i in range(runs)]
            sim_results = pool.starmap(simulate_one_scenario, args)

    results = {'overlap': [res['overlap'] for res in sim_results]}
    plot_overlap(results['overlap'], baseline=centrality1, centrality=centrality2, zoom=False, note=note)


import sys

def main():

    if len(sys.argv) > 1:
        all_conf_files = sys.argv[1:]
    else:
        all_conf_files = ['example_config.ini']

    for conf_file in all_conf_files:
        configuration = library.get_config_from_ini(conf_file)

        cached_files = None
        if configuration.cache:
            print("Caching enabled. Preparing graphs...")
            # 1. Identify all required centralities
            all_centralities = set()
            for pair in configuration.centralities:
                all_centralities.add(pair[0])
                all_centralities.add(pair[1])
            all_centralities = list(all_centralities)

            # 2. Get existing files
            existing_files = library.get_cached_files(configuration)
            print(f"Found {len(existing_files)} existing files in cache.")

            # 3. Process graphs (Load/Generate -> Compute -> Save)
            # We need 'runs' graphs.
            # If runs > existing, we generate new ones.
            # If runs <= existing, we use the first 'runs' files (or all? usually runs).

            process_args = [(i, configuration, existing_files, all_centralities) for i in range(configuration.runs)]

            if configuration.max_processes == 1:
                cached_files = [library.process_graph(*args) for args in process_args]
            else:
                with multiprocessing.Pool(processes=configuration.max_processes) as pool:
                    cached_files = pool.starmap(library.process_graph, process_args)

            print(f"Prepared {len(cached_files)} graphs.")

        if configuration.only_compute:
            continue

        for centrality_pair in configuration.centralities:
            run_comparing_sim(configuration.runs, centrality_pair[0], centrality_pair[1], configuration, note=configuration.note, parallel=(configuration.max_processes != 1), cached_files=cached_files)
            print(f"Completed simulation for {centrality_pair[0]} vs. {centrality_pair[1]}, configuration: {conf_file}")


if __name__ == "__main__":
    main()
