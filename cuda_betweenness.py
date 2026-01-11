import cupy as cp
import rmm

free_mem, total_mem = cp.cuda.runtime.memGetInfo()

initial_pool = int(0.80 * total_mem) - (int(0.80 * total_mem) % 256)
max_pool = int(0.99 * total_mem) - (int(0.99 * total_mem) % 256)

rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=initial_pool,
    maximum_pool_size=max_pool,
)

import sys
import os
import glob
import pickle
import traceback
import numpy as np
import networkx as nx
import library

# Check for cugraph availability
try:
    # We don't import cugraph directly here because we use the networkx backend.
    # However, to fail fast, we can check if it's importable.
    import cugraph
    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False

def print_debug_info():
    print("=== Debug Info ===")
    print(f"NetworkX version: {nx.__version__}")
    print(f"NumPy version: {np.__version__}")

    try:
        import cugraph
        print(f"cugraph version: {cugraph.__version__}")
    except ImportError:
        print("cugraph not installed")

    try:
        import cupy
        print(f"cupy version: {cupy.__version__}")
        print(f"CUPY_CACHE_DIR: {os.environ.get('CUPY_CACHE_DIR', 'Not Set')}")
    except ImportError:
        print("cupy not installed")

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print("==================")

def smoke_test():
    print("=== Running Smoke Test ===")
    if not HAS_CUGRAPH:
        print("Skipping smoke test (cugraph not detected)")
        return

    try:
        print("Creating small test graph...")
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        print("Running betweenness_centrality with backend='cugraph'...")
        bc = nx.betweenness_centrality(G, backend="cugraph")
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
            return

        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

        # To compute betweenness we explicitly request the cugraph backend
        scores_dict = nx.betweenness_centrality(graph, backend="cugraph")

        # Faster conversion: Convert to sorted array
        if graph.number_of_nodes() > 0:
            # We assume nodes are 0..N-1
            scores_arr = np.array([scores_dict[i] for i in range(graph.number_of_nodes())])
        else:
            scores_arr = np.array([])

        np.save(npy_path, scores_arr)
        print(f"Computed betweenness for {base_name}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        # Specific check for the malformed CompileException
        if "CompileException.__init__() missing" in str(e):
             print("\n!!! DIAGNOSIS: It appears a CUDA compilation error occurred, but the exception was malformed.")
             print("This often indicates a mismatch between cupy/cugraph versions or a CUDA environment issue.")
             print("Please check your CUDA installation and cupy/cugraph compatibility.\n")
        # Abort on first failure
        raise e

def main():
    # GPU Selection: Respect env var if set, otherwise default to device 0
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("CUDA_VISIBLE_DEVICES not set. Defaulting to device 0.")
    else:
        print(f"Using defined CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    print_debug_info()
    smoke_test()

    if not HAS_CUGRAPH:
        print("Error: cugraph module not found. Please install cugraph and ensure dependencies are met.")
        sys.exit(1)

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

        files = library.get_cached_files(config)

        if not files:
            print(f"No cached graph files found for {conf_file}")
            continue

        print(f"Found {len(files)} graph files.")

        for file_path in files:
            process_graph(file_path, config.cache_dir)

if __name__ == "__main__":
    main()
