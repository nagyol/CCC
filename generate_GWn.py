import sys
import os
import uuid
import pickle
import random
import multiprocessing
import argparse
import time
import numpy as np
import networkx as nx

import library

def _generate_chunk(seed, n, latents, c, model_type, start_idx, end_idx, directed):
    """
    Worker function to generate edges for a chunk of rows.
    """
    rng = np.random.default_rng(seed)

    edges = []

    row_latents = latents[start_idx:end_idx]

    for i_local, u_i in enumerate(row_latents):
        i = start_idx + i_local

        if model_type.endswith("GWnA"):
            # W = c * u_i * u_j
            probs = c * u_i * latents
        elif model_type.endswith("GWnB"):
            # W = c * (u_i + u_j)
            probs = c * (u_i + latents)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if np.any(probs > 1.0):
             probs = np.minimum(probs, 1.0)

        r = rng.random(n)

        if directed:
            #  excluded self-loops
            mask = (r < probs)
            mask[i] = False # No self loops
            neighbors = np.where(mask)[0]

            for j in neighbors:
                edges.append((i, j))

        else:
            mask = (r < probs)

            if i + 1 < n:
                valid_probs = probs[i+1:]
                valid_r = r[i+1:]
                local_mask = valid_r < valid_probs
                # local_mask indices 0 correspond to j = i+1
                local_neighbors = np.where(local_mask)[0]
                neighbors = local_neighbors + (i + 1)

                for j in neighbors:
                    edges.append((i, j))

    return edges

def generate_gwn_graph(config, run_index):
    """
    Generates a single GWn graph based on config.
    """
    N = config.N
    M = config.M

    c = config.c
    if c is None:
        if "GWnA" in M:
            c = 1.0
        elif "GWnB" in M:
            c = 0.5
        else:
            c = 1.0 # Fallback

    directed = M.startswith("directed")

    print(f"Generating run {run_index+1}/{config.runs}: N={N}, Model={M}, c={c}, Directed={directed}")

    latents = np.random.uniform(0, 1, N)

    edges = []

    PARALLEL_THRESHOLD = 1000

    if N > PARALLEL_THRESHOLD:
        print(f"  Using parallel edge generation (N={N} > {PARALLEL_THRESHOLD})...")
        num_workers = multiprocessing.cpu_count()
        chunk_size = (N + num_workers - 1) // num_workers

        tasks = []
        for worker_id in range(num_workers):
            start = worker_id * chunk_size
            end = min(start + chunk_size, N)
            if start >= end:
                break

            worker_seed = random.randint(0, 2**32 - 1)

            tasks.append((worker_seed, N, latents, c, M, start, end, directed))

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(_generate_chunk, tasks)

        for res in results:
            edges.extend(res)

    else:
        seed = random.randint(0, 2**32 - 1)
        edges = _generate_chunk(seed, N, latents, c, M, 0, N, directed)

    print(f"  Constructing graph with {len(edges)} edges...")
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    # Naming convention: graph_{M}_{N}_cent_{suffix}_{uuid}.gpickle
    unique_id = uuid.uuid4().hex
    filename = f"graph_{M}_{N}_cent_{config.suffix}_{unique_id}.gpickle"

    cache_dir = config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(cache_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(G, f)

    print(f"  Saved to {filepath}")


def main():
    if len(sys.argv) > 1:
        ini_file = sys.argv[1]
    else:
        ini_file = "example_config.ini"

    if not os.path.exists(ini_file):
        print(f"Error: Configuration file '{ini_file}' not found.")
        sys.exit(1)

    print(f"Reading configuration from {ini_file}...")
    try:
        config = library.get_config_from_ini(ini_file)
    except Exception as e:
        print(f"Error parsing config: {e}")
        sys.exit(1)

    valid_types = ["directed-GWnA", "GWnA", "directed-GWnB", "GWnB"]
    if config.M not in valid_types:
        print(f"Graph model '{config.M}' is not a GWn type. Skipping generation.")
        print(f"This script only handles: {', '.join(valid_types)}")
        sys.exit(0)

    start_time = time.time()
    for i in range(config.runs):
        generate_gwn_graph(config, i)

    elapsed = time.time() - start_time
    print(f"Done. Generated {config.runs} graphs in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
