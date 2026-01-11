import os
import sys
import pickle
import traceback
import numpy as np
import networkx as nx
import joblib
import nx_parallel
import library

def print_debug_info():
    print("=== Debug Info ===")
    print(f"NetworkX version: {nx.__version__}")
    print(f"nx_parallel version: {nx_parallel.__version__}")
    print(f"NumPy version: {np.__version__}")

    cpu_count = os.cpu_count()
    print(f"CPU count: {cpu_count}")

    n_jobs = max(1, cpu_count - 1) if cpu_count else 1
    print(f"Using n_jobs: {n_jobs}")
    print("==================")

def smoke_test():
    print("=== Running Smoke Test ===")
    try:
        print("Creating small test graph...")
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        cpu_count = os.cpu_count()
        n_jobs = max(1, cpu_count - 1) if cpu_count else 1

        print(f"Running betweenness_centrality with backend='parallel' (n_jobs={n_jobs})...")
        with joblib.parallel_config(n_jobs=n_jobs):
            bc = nx.betweenness_centrality(G, backend="parallel")
        print(f"Smoke test passed. Result: {bc}")
    except Exception:
        print("Smoke test FAILED with exception:")
        traceback.print_exc()
    print("==========================")

def process_graph(file_path, cache_dir):
    """
    Process a single graph file.
    Aborts immediately on failure.
    """
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        npy_filename = f"{base_name}_centrality_betweenness.npy"
        npy_path = os.path.join(cache_dir, npy_filename)

        if os.path.exists(npy_path):
            # Resumable: skip if exists
            return

        # Load graph
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

        # Compute betweenness
        cpu_count = os.cpu_count()
        n_jobs = max(1, cpu_count - 1) if cpu_count else 1

        with joblib.parallel_config(n_jobs=n_jobs):
            scores_dict = nx.betweenness_centrality(graph, backend="parallel")

        # Convert to sorted array
        if graph.number_of_nodes() > 0:
            # We assume nodes are 0..N-1
            scores_arr = np.array([scores_dict[i] for i in range(graph.number_of_nodes())])
        else:
            scores_arr = np.array([])

        np.save(npy_path, scores_arr)
        print(f"Computed betweenness for {base_name}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        # Abort on first failure
        raise e

def main():
    print_debug_info()
    smoke_test()

    if len(sys.argv) > 1:
        conf_files = sys.argv[1:]
    else:
        conf_files = ['example_config.ini']

    for conf_file in conf_files:
        print(f"Processing config: {conf_file}")
        try:
            config = library.get_config_from_ini(conf_file)
        except Exception as e:
            print(f"Error loading config {conf_file}: {e}")
            sys.exit(1)

        # Discover files
        files = library.get_cached_files(config)

        if not files:
            print(f"No cached graph files found for {conf_file}")
            continue

        print(f"Found {len(files)} graph files.")

        for file_path in files:
            process_graph(file_path, config.cache_dir)

if __name__ == "__main__":
    main()
