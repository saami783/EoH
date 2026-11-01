import os
import sys
import time
from typing import Tuple, Any
from pathlib import Path

from helpers.results import aggregate_and_log_results, PROB_METRICS_FILE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import signal
from eoh.eoh.eoh import EOH
import threading
from eoh.llm.interface_LLM import InterfaceLLM
from eoh import EoH, LLMConfig
from prob import ConjectureRefutation
import argparse

from helpers.database import create_db
from helpers.subclass import subclass_function

_DEFAULT_MUTATION_NAMES_FALLBACK: Tuple[str, ...] = (
    "add_edge",
    "remove_edge",
    "add_vertex",
    "remove_vertex",
    "subdivision",
    "contraction",
    "replace_vertex_by_path",
    "replace_vertex_by_star",
    "replace_vertex_by_clique",
    "replace_vertex_by_polyhedral",
    "bipartition_neighborhood",
)

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Automated graph theory conjecture refutation program.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("--script", type=str, default="None", help="Path to a Python script containing a scoring function of the form score(G, min_size, max_size).")
    parser.add_argument("--function", type=str, default="None", help="Name of the scoring function to load from the script specified via --script.")
    parser.add_argument("--min-order", type=int, default=6, help="Minimum number of vertices of the tested graphs.")
    parser.add_argument("--max-order", type=int, default=30, help="Maximum number of vertices of the tested graphs.")

    parser.add_argument("--time-limit", type=float, default=60.0, help="Maximum time in seconds allocated to the search.")
    parser.add_argument("--timeout", type=int, default=30,help="Maximum time allowed for the LLM-generated code to execute on a graph.")

    parser.add_argument("--local-llm", action="store_true", default=False, help="Use a local LLM or via API request (OpenAI etc.). This parameter adjusts the number of concurrent requests sent: 2 requests for a local LLM vs. 4 for a hosted LLM.")
    parser.add_argument("--llm", choices=["ollama", "gemini", "mistral", "codex", "deepseek", "gpt"], default=None, help="LLM backend used by EoH.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")
    parser.add_argument("--neighbors", type=int, default=20, help="Number of neighbors explored per iteration for the local search.")
    parser.add_argument("--max-mutations", type=int, default=2, help="Maximum number of mutations applied to build a neighbor.")
    parser.add_argument("--stagnation", type=int, default=10, help="Number of iterations without improvement before a restart.")
    parser.add_argument("--margin", type=float, default=1e-3, help="Numerical margin required to accept a counterexample.")
    parser.add_argument("--cache-limit", type=int, default=None, help="Maximum number of evaluations kept in cache.")
    parser.add_argument("--cpu", type=int, default=max(1, os.cpu_count()), help="Number of worker processes; <= 1 disables multiprocessing.")
    parser.add_argument("--np-hard", action="store_true", help="Compute a list of NP-hard invariants to guide the LLM.")
    parser.add_argument("--subclass", default=None, choices=list(subclass_function.keys()), help="Target graph subclass for the conjecture, e.g., 'planar', 'connected, claw_free', 'bipartite'. If empty, no subclass restriction is applied. The graph generated for the search will be empty.")
    parser.add_argument("--use-db", action="store_true", default=False, help="Use a database of counterexamples to force FunSearch to find unique counterexamples for the conjectures.")
    parser.add_argument("--beam-width", type=int, default=5, help="Number of graphs kept at each step.")
    parser.add_argument("--max-steps", type=int, default=15, help="Number of mutation iterations.")

    # parser.add_argument("--approx", type=bool, action="store_true", default=False, help="Two-stage evaluation: the 'approx(G, min_size, max_size)' function is used first for evaluation; if the score is negative, the true scoring function will be used on the graph.")
    # parser.add_argument(
    #     "--mutations",
    #     nargs="+",
    #     default=
    #         ["add_edge",
    #         "remove_edge",
    #         "add_vertex",
    #         "remove_vertex",
    #         "subdivision",
    #         "contraction",
    #         "replace_vertex_by_path",
    #         "replace_vertex_by_star",
    #         "replace_vertex_by_clique",
    #         "replace_vertex_by_polyhedral",
    #         "bipartition_neighborhood"],
    #     help="Allowed mutation operators during the search."
    # )
    args = parser.parse_args()
    return args


def _verify_args(run_params: dict[str, Any]):
    if run_params["script"] == "None" or run_params["function"] ==  "None":
        raise Exception("You must define both the path to the scoring function and the name of the scoring function.")
    if run_params["min_order"] > run_params["max_order"]:
        raise Exception("The minimum order of the graph cannot be greater than the maximum order.")
    if run_params["timeout"] > run_params["time_limit"]:
        raise Exception("The execution time allowed for the LLM on a graph cannot be greater than the program execution time.")

    script_path = run_params["script"]
    if not script_path.endswith('.py'):
        script_path += '.py'

    actual_path = os.path.join(BASE_DIR, script_path)

    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"File {actual_path} not found.")

    if run_params["use_db"]:
        create_db()


def _init_json_files():
    Path("counter.json").write_text("{}", encoding="utf-8")
    Path("counter.json.lock").unlink(missing_ok=True)
    PROB_METRICS_FILE.unlink(missing_ok=True)


# Pour arrêter le programme prématurément
_orig_record = EOH._record

def _stop_when_negative(self, op: str, offspring) -> bool:
  is_new_best = _orig_record(self, op, offspring)

  if offspring and offspring.get("objective") is not None and offspring["objective"] < 0:
    self._flush_samples()
    os.kill(os.getpid(), signal.SIGINT)
  return is_new_best

EOH._record = _stop_when_negative


# Pour compter le vrai nombre de requêtes api utilisées sans la phase de test
_api_lock = threading.Lock()
api_request_count = 0

_orig_get_response = InterfaceLLM.get_response

def _counted_get_response(self, prompt_content: str) -> str:
    global api_request_count
    with _api_lock:
        api_request_count += 1
    return _orig_get_response(self, prompt_content)

InterfaceLLM.get_response = _counted_get_response

def main():
    args = parse_args()

    params = {
        "script": args.script,
        "function": args.function,
        "min_order": args.min_order,
        "max_order": args.max_order,
        "time_limit": args.time_limit,
        "seed": args.seed,
        "local_llm": args.local_llm,
        "timeout": args.timeout,
        "neighbors": args.neighbors,
        "max_mutations": args.max_mutations,
        "stagnation": args.stagnation,
        "margin": args.margin,
        "cache_limit": args.cache_limit,
        "cpu": args.cpu,
        "llm": args.llm,
        "np_hard": args.np_hard,
        "subclass": args.subclass,
        "use_db": args.use_db,
        "beam_width": args.beam_width,
        "max_steps": args.max_steps,
        # "mutations": args.mutations,
    }

    _verify_args(params)

    task = ConjectureRefutation(
        min_order=args.min_order,
        max_order=args.max_order,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        timeout=args.timeout,
        cpu=args.cpu,
        conjecture_score_function_path=args.script,
        score_function_name=args.function,
        used_counterexamples_db=args.use_db,
        subclass=args.subclass
    )

    if args.llm == "ollama":
        llm = LLMConfig(
            use_local=True,
            local_url=os.environ.get("EOH_OLLAMA_URL", "http://192.168.1.XX:8000/api/chat/ollama"),
            timeout=300,
        )
    elif args.llm == "codex":
        llm = LLMConfig(
            use_local=True,
            local_url=os.environ.get("EOH_CODEX_URL", "http://192.168.1.XX:8000/api/chat/codex"),
            timeout=600,
        )
    else:
        llm = LLMConfig(
            api_endpoint=os.environ.get("EOH_API_ENDPOINT", "api.deepseek.com"),
            api_key=os.environ.get("EOH_API_KEY", "sk-xxx"),
            model=os.environ.get("EOH_MODEL", "deepseek-chat"),
            timeout=150,
        )

    eoh = EoH(
        llm=llm,
        problem=task,
        num_samplers=4,
        num_evaluators=4,
        pop_size=10,
        n_pop=20,
        operators=['e1', 'e2', 'm1', 'm2', 'm3'],
        output_dir=os.path.dirname(__file__),
    )

    _init_json_files()

    start_time = time.perf_counter()

    try:
        eoh.run()
    except KeyboardInterrupt:
        print("")
    finally:
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        print(f"Execution time : {elapsed_seconds:.6f} seconds")
        print(f"Count api requests : {api_request_count}")

        aggregate_and_log_results(
            params=params,
            elapsed_time=elapsed_seconds,
            api_requests=api_request_count,
        )

if __name__ == "__main__":
    main()