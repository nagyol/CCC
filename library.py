import datetime
import glob
import multiprocessing
import os
import pickle
import uuid
import typing
from collections import namedtuple
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
from itertools import combinations
from pathlib import Path
from os import path

import networkx as nx
import numpy as np

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

Conf = namedtuple('Conf', 'generator note name centralities suffix runs cache save load cache_dir N M only_compute max_processes cutoff c')

def nx_to_igraph(graph: typing.Union[nx.Graph, nx.DiGraph]) -> "ig.Graph":
    if not HAS_IGRAPH:
        raise ImportError("igraph not installed")

    if nx.is_frozen(graph) and hasattr(graph, "_cached_igraph"):
        return graph._cached_igraph

    ig_graph = ig.Graph.from_networkx(graph)

    if nx.is_frozen(graph):
        try:
            graph._cached_igraph = ig_graph
        except AttributeError:
            pass

    return ig_graph

def _igraph_centrality_wrapper(graph: nx.Graph, func_name: str, **kwargs) -> typing.Dict:
    if HAS_IGRAPH:
        try:
            ig_graph = nx_to_igraph(graph)
            if func_name == "pagerank":
                scores = ig_graph.pagerank(**kwargs)
            elif func_name == "betweenness":
                scores = ig_graph.betweenness(**kwargs)
            elif func_name == "closeness":
                scores = ig_graph.closeness(normalized=True, **kwargs)
            elif func_name == "eigenvector_centrality":
                scores = ig_graph.eigenvector_centrality(**kwargs)
            elif func_name == "degree":
                scores = ig_graph.degree(**kwargs)
            elif func_name == "indegree":
                 scores = ig_graph.degree(mode="in", **kwargs)
            elif func_name == "outdegree":
                 scores = ig_graph.degree(mode="out", **kwargs)
            elif func_name == "harmonic_centrality":
                 if hasattr(ig_graph, "harmonic_centrality"):
                     scores = ig_graph.harmonic_centrality(**kwargs)
                 else:
                     return None
            else:
                return None

            if "_nx_name" in ig_graph.vs.attributes():
                return {v["_nx_name"]: s for v, s in zip(ig_graph.vs, scores)}
            else:
                return {i: s for i, s in enumerate(scores)}
        except Exception as e:
            print(f"igraph failed for {func_name}: {e}. Falling back to NetworkX.")
            pass
    return None

def get_config_from_ini(path_to_ini: str) -> typing.NamedTuple:
    config_object = ConfigParser(interpolation=ExtendedInterpolation())
    config_object.read(path_to_ini)

    topology = config_object["TOPOLOGY"]
    vertex_count = int(topology["N"])
    graph_model = topology["M"]
    gamma = None
    out_gamma = None
    note = graph_model
    if graph_model == "configuration-model":
        gamma = float(config_object["CONFIGURATIONMODEL"]["gamma"])
        note = f'{graph_model}-"PL_exp":{gamma}'
    if graph_model == "directed-CM":
        gamma = float(config_object["CONFIGURATIONMODEL"]["gamma"])
        out_gamma = float(config_object["CONFIGURATIONMODEL"]["out_gamma"])
        note = f'{graph_model}-"inPL_exp":{gamma}-"outPL_exp":{out_gamma}'

    simulation = config_object["SIMULATION"]
    cutoff_val = simulation.get("cutoff", fallback=None)
    if cutoff_val is not None:
        cutoff_val = int(cutoff_val)

    c_val = simulation.get("c", fallback=None)
    if c_val is not None:
        c_val = float(c_val)

    if simulation["centralities"] == "pagerankDF":
        centralities = []
        if "DFstep" in simulation.keys():
            df_step = int(simulation["DFstep"])
        else:
            df_step = 10
        for i in range(0, 100, df_step):
            centralities.append(f"pagerank-{i}")
            all_scenarios = list(combinations(centralities, 2))
    else:
        raw_centralities = [x.strip() for x in simulation["centralities"].split(',')]
        centralities = []
        for c in raw_centralities:
            if c == "betweenness1-Q":
                if "Q" in simulation:
                    Q_val = int(simulation["Q"])
                    for i in range(1, Q_val + 1):
                        centralities.append(f"betweenness-{i}")
                else:
                    print("Warning: betweenness1-Q requested but Q not defined in SIMULATION.")
            elif c == "closeness1-Q":
                if "Q" in simulation:
                    Q_val = int(simulation["Q"])
                    for i in range(1, Q_val + 1):
                        centralities.append(f"closeness-{i}")
                else:
                    print("Warning: closeness1-Q requested but Q not defined in SIMULATION.")
            else:
                if cutoff_val is not None:
                    if c == "betweenness":
                        c = f"betweenness-{cutoff_val}"
                    elif c == "harmonic":
                        c = f"harmonic-{cutoff_val}"
                    elif c == "closeness":
                        c = f"closeness-{cutoff_val}"
                    elif c == "load":
                        c = f"load-{cutoff_val}"
                centralities.append(c)
        all_scenarios = list(combinations(centralities, 2))
    runs = int(simulation["runs"])

    suffix = simulation.get("suffix", "")
    cache = simulation.getboolean("cache", fallback=False)
    save = simulation.getboolean("save", fallback=False)
    load = simulation.getboolean("load", fallback=False)
    cache_dir = simulation.get("cache_dir", "cache")

    graph_generator = partial(generate_network, vertex_count, graph_model, gamma, out_gamma)
    full_note = f'{note}{("" if suffix == "" else "-")}{suffix}'

    only_compute = simulation.getboolean("only_compute", fallback=False)
    max_processes = simulation.get("max_processes", fallback=None)
    if max_processes is not None:
        max_processes = int(max_processes)

    return Conf(graph_generator, full_note, path_to_ini, all_scenarios, suffix, runs, cache, save, load, cache_dir, vertex_count, graph_model, only_compute, max_processes, cutoff_val, c_val)

def generate_network(n, net_type, gamma=None, out_gamma=None):
    match net_type:
        case "directed-GWnA" | "GWnA" | "directed-GWnB" | "GWnB":
            raise RuntimeError(f"Graph type '{net_type}' cannot be generated internally. Please use 'generate_GWn.py' to generate the graph files first.")
        case "scale-free":
            graph = nx.powerlaw_cluster_graph(n, 5, 0.3)
            while not nx.is_connected(graph):
                graph = nx.powerlaw_cluster_graph(n, 5, 0.3)
        case "small-world":
            graph = nx.newman_watts_strogatz_graph(n, 6, 0.6)
            while not nx.is_connected(graph):
                graph = nx.newman_watts_strogatz_graph(n, 6, 0.6)
        case "Erdos-Renyi":
            graph = nx.fast_gnp_random_graph(n, 0.01)
            while not nx.is_connected(graph):
                graph = nx.fast_gnp_random_graph(n, 0.01)
        case "barabasi-albert":
            graph = nx.barabasi_albert_graph(n, 1)
        case "hep-th":
            graph = nx.read_edgelist('Cit-HepTh.txt')
        case "coll-grqc-LCC":
            tmp_graph = nx.read_edgelist('CA-GrQc.txt')
            tmp_graph = tmp_graph.subgraph(max(nx.connected_components(tmp_graph), key=len))
            graph = nx.convert_node_labels_to_integers(tmp_graph)
        case "coll-grqc":
            tmp_graph = nx.read_edgelist('CA-GrQc.txt')
            graph = nx.convert_node_labels_to_integers(tmp_graph)
        case "cit-hepph":
            tmp_graph = nx.read_edgelist('Cit-HepPh.txt', create_using=nx.DiGraph)
            graph = nx.convert_node_labels_to_integers(tmp_graph)
        case "configuration-model":
            while True:
                degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
                if nx.is_graphical(degree_sequence):
                    break
            graph = nx.configuration_model(degree_sequence)
        case "configuration-model-simple":
            while True:
                degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
                if nx.is_graphical(degree_sequence):
                    break
            graph = nx.configuration_model(degree_sequence)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        case "directed-CM":
            in_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
            while True:
                out_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not out_gamma else out_gamma)]
                if sum(in_degree_sequence) == sum(out_degree_sequence):
                    break
            graph = nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
        case "directed-CM-simple":
            in_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not gamma else gamma)]
            while True:
                out_degree_sequence = [int(d) + 1 for d in nx.utils.powerlaw_sequence(n, 3 if not out_gamma else out_gamma)]
                if sum(in_degree_sequence) == sum(out_degree_sequence):
                    break
            graph = nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
            graph = nx.DiGraph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        case _:
            raise Exception("Missing network type")
    return graph

def get_ranking_pagerank(graph: nx.Graph, alpha: float = 0.85) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "pagerank", damping=alpha)
    if res is not None: return res
    return nx.algorithms.pagerank(graph, alpha=alpha)

def get_ranking_harmonic(graph: nx.Graph, cutoff: int = None) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "harmonic_centrality", cutoff=cutoff)
    if res is not None: return res
    return nx.algorithms.harmonic_centrality(graph)

def get_ranking_eigenvector(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "eigenvector_centrality", scale=False)
    if res is not None: return res
    # Use iterative fallback to avoid OOM on large graphs
    return nx.algorithms.eigenvector_centrality(graph)

def get_ranking_betweenness(graph: nx.Graph, cutoff: int = None) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "betweenness", cutoff=cutoff)
    if res is not None: return res
    if cutoff is not None:
        print(f"Warning: NetworkX does not support cutoff for betweenness. Ignoring cutoff={cutoff}.")
    return nx.algorithms.betweenness_centrality(graph)

def get_ranking_load(graph: nx.Graph, cutoff: int = None) -> typing.Dict:
    return nx.algorithms.load_centrality(graph, cutoff=cutoff)

def get_ranking_degree(graph: nx.Graph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "degree")
    if res is not None: return res
    return nx.algorithms.degree_centrality(graph)

def get_ranking_indegree(graph: nx.DiGraph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "indegree")
    if res is not None: return res
    return nx.algorithms.in_degree_centrality(graph)

def get_ranking_outdegree(graph: nx.DiGraph) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "outdegree")
    if res is not None: return res
    return nx.algorithms.out_degree_centrality(graph)

def get_ranking_closeness(graph: nx.Graph, cutoff: int = None) -> typing.Dict:
    res = _igraph_centrality_wrapper(graph, "closeness", cutoff=cutoff)
    if res is not None: return res
    if cutoff is not None:
        print(f"Warning: NetworkX does not support cutoff for closeness. Ignoring cutoff={cutoff}.")
    return nx.algorithms.closeness_centrality(graph)

def get_ranking_katz(graph: nx.Graph) -> typing.Dict:
    if graph.is_multigraph():
        graph = nx.DiGraph(graph)
    alpha = 1./(2*max(graph.degree())[0])
    # Use iterative solver to avoid OOM on large graphs (numpy version is dense)
    return nx.algorithms.katz_centrality(graph, alpha=alpha)

def get_ranking(centrality: typing.AnyStr):
    lookup = {
        "betweenness": get_ranking_betweenness,
        "closeness": get_ranking_closeness,
        "harmonic": get_ranking_harmonic,
        "pagerank": get_ranking_pagerank,
        "degree": get_ranking_degree,
        "load": get_ranking_load,
        "katz": get_ranking_katz,
        "eigenvector": get_ranking_eigenvector,
        "indegree": get_ranking_indegree,
        "outdegree": get_ranking_outdegree
    }
    for damping_factor in range(0,100):
        lookup.update({f"pagerank-{damping_factor}": partial(get_ranking_pagerank, alpha=damping_factor/100)})

    if centrality in lookup:
        return lookup[centrality]

    # Dynamic parsing for betweenness-Y and closeness-Y
    if centrality.startswith("betweenness-"):
        try:
            cutoff = int(centrality.split("-")[1])
            return partial(get_ranking_betweenness, cutoff=cutoff)
        except ValueError:
            pass

    if centrality.startswith("closeness-"):
        try:
            cutoff = int(centrality.split("-")[1])
            return partial(get_ranking_closeness, cutoff=cutoff)
        except ValueError:
            pass

    if centrality.startswith("load-"):
        try:
            cutoff = int(centrality.split("-")[1])
            return partial(get_ranking_load, cutoff=cutoff)
        except ValueError:
            pass

    if centrality.startswith("harmonic-"):
        try:
            cutoff = int(centrality.split("-")[1])
            return partial(get_ranking_harmonic, cutoff=cutoff)
        except ValueError:
            pass

    return lookup[centrality]

def run_in_parallel(runs: int, fn: typing.Callable) -> typing.List:
    with multiprocessing.Pool() as pool:
        async_results = [pool.apply_async(fn) for _ in range(runs)]
        results = [result.get() for result in async_results]
    return results

def get_cached_files(config: Conf) -> typing.List[str]:
    # Ensure cache directory exists
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # Construct search pattern
    # Pattern: graph_{M}_{N}_cent_{suffix}_{index/uuid}.gpickle
    # Note: suffix can be empty.

    prefix = f"graph_{config.M}_{config.N}_cent_{config.suffix}_"
    pattern = path.join(config.cache_dir, f"{prefix}*.gpickle")

    files = glob.glob(pattern)
    files.sort() # Ensure deterministic order
    return files

def process_graph(index: int, config: Conf, cached_files: typing.List[str], all_centralities: typing.List[str]) -> str:

    file_path = None
    graph = None
    modified_graph = False

    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    if config.load and index < len(cached_files):
        file_path = cached_files[index]
        try:
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}. Generating new graph.")
            graph = None

    if graph is None:
        graph = config.generator()
        modified_graph = True
        unique_id = uuid.uuid4().hex
        filename = f"graph_{config.M}_{config.N}_cent_{config.suffix}_{unique_id}.gpickle"
        file_path = path.join(config.cache_dir, filename)

    if not nx.is_frozen(graph):
        nx.freeze(graph)

    # Base name for centrality files: matches graph filename but with .npy extension and suffix
    base_name = path.splitext(path.basename(file_path))[0]

    for cent_name in all_centralities:
        npy_filename = f"{base_name}_centrality_{cent_name}.npy"
        npy_path = path.join(config.cache_dir, npy_filename)

        if not path.exists(npy_path):
            ranking_func = get_ranking(cent_name)
            scores_dict = ranking_func(graph)

            # Convert to array assuming nodes are 0..N-1
            # If nodes are exactly range(N):
            if graph.number_of_nodes() > 0:
                n_nodes = graph.number_of_nodes()
                keys = np.fromiter(scores_dict.keys(), dtype=int, count=n_nodes)
                vals = np.fromiter(scores_dict.values(), dtype=float, count=n_nodes)
                scores_arr = np.empty(n_nodes)
                scores_arr[keys] = vals
            else:
                scores_arr = np.array([])

            np.save(npy_path, scores_arr)

    # Save graph topology if needed (only if new or modified)
    if config.save and modified_graph:
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)

    return file_path

def _get_normalized_total_orderings_arrays(baseline: typing.Union[typing.List, typing.Dict], other: typing.Union[typing.List, typing.Dict]) -> typing.List[np.ndarray]:
    n = len(baseline)
    if isinstance(baseline, dict):
        keys = np.fromiter(baseline.keys(), dtype=int, count=n)
        vals = np.fromiter(baseline.values(), dtype=float, count=n)
        baseline_arr = np.empty(n)
        baseline_arr[keys] = vals
    else:
        baseline_arr = np.asarray(baseline)

    if isinstance(other, dict):
        keys = np.fromiter(other.keys(), dtype=int, count=n)
        vals = np.fromiter(other.values(), dtype=float, count=n)
        other_arr = np.empty(n)
        other_arr[keys] = vals
    else:
        other_arr = np.asarray(other)

    random_arr = np.random.uniform(size=n)
    indices = np.arange(n)

    keys_baseline = (-random_arr, -other_arr, -baseline_arr)
    normalized_baseline = indices[np.lexsort(keys_baseline)]

    keys_other = (-random_arr, -baseline_arr, -other_arr)
    normalized_other = indices[np.lexsort(keys_other)]

    return [normalized_baseline, normalized_other]

def get_normalized_total_orderings(baseline: typing.Union[typing.List, typing.Dict], other: typing.Union[typing.List, typing.Dict]) -> typing.List:
    norm_arrays = _get_normalized_total_orderings_arrays(baseline, other)
    return [arr.tolist() for arr in norm_arrays]

def compare_centrality(baseline: typing.List, ranking_other: typing.List) -> typing.Dict:
    norm_baseline, norm_other = _get_normalized_total_orderings_arrays(baseline, ranking_other)
    n = len(baseline)

    inv_nb = np.zeros(n, dtype=int)
    inv_nb[norm_baseline] = np.arange(n)

    inv_no = np.zeros(n, dtype=int)
    inv_no[norm_other] = np.arange(n)

    max_ranks = np.maximum(inv_nb, inv_no)

    counts = np.bincount(max_ranks, minlength=n)
    overlap_list = np.cumsum(counts)

    return {'overlap': overlap_list.tolist()}

def plot_general(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, header: typing.AnyStr = None,
                 parabola: bool = False, zoom: bool = True, rescale: bool = False, note: typing.AnyStr = None) -> None:
    # Delayed import to avoid issues with multiprocessing and backend configuration
    import matplotlib
    # Force Agg backend for headless environments/clusters
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if isinstance(results[0], list):
        mean = np.mean(np.array(results), axis=0)
        error = np.std(np.array(results), axis=0)
    else:
        mean = results
        error = 0

    N = len(mean)
    mean = mean / N
    if isinstance(error, np.ndarray) or error > 0:
        error = error / N
    x_vals = np.arange(1, N + 1) / N

    fig, ax = plt.subplots()

    ax.errorbar(x_vals, mean, yerr=(error if len(results) > 1 else None), ecolor='r', capsize=5,
                 linestyle='none', marker='x',
                 markersize=2)

    if zoom:
        ax.set_title(
            f"{header} {baseline} vs. {centrality} \n Runs:{len(results)}, graph size: {N}, top 10% of vertices \n {note}")
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 0.1)

        ticks = np.arange(0, 0.1001, 0.02)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        fig.savefig(f"{datetime.datetime.now()}-{baseline}-{centrality}-top10p.pdf")

    ax.set_title(
        f"{header} {baseline} vs. {centrality} \n Runs:{len(results)}, graph size: {N}, all vertices  \n {note}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ticks = np.arange(0, 1.001, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Add diagonal line
    ax.plot([0, 1], [0, 1], ls="--", c=".3")

    if parabola:
        points = np.linspace(0, 1, 100000)
        y_points = points ** 2
        ax.plot(points, y_points)

    Path("./results").mkdir(parents=True, exist_ok=True)
    filename = fr"{baseline}-{centrality}-{datetime.datetime.now():%Y-%m-%d-%H%M%S}.pdf"
    destination = path.join(path.abspath("results"), filename)
    fig.savefig(destination)

    plt.close(fig)
