"""
AI Research Agent CLI - Ollama Local Version

This is the main entry point for the AI Research Agent.
Supports both single-agent and orchestrator modes, running 100% locally with Ollama.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

from agent import run_experiment_loop
from logger import print_status


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Research Agent CLI - Runs locally with Ollama"
    )
    parser.add_argument(
        "task",
        type=str,
        help=(
            "In 'single' mode: the hypothesis to verify.\n"
            "In 'orchestrator' mode: the high-level research task to investigate."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU info (informational only, execution is local).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "orchestrator"],
        default="single",
        help=(
            "Execution mode: "
            "'single' runs a single-researcher agent; "
            "'orchestrator' runs the multi-agent orchestrator."
        ),
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="(orchestrator) Number of initial hypotheses to generate.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="(orchestrator) Maximum number of orchestration rounds.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="(orchestrator) Maximum parallel experiments.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with mock data (no LLM usage).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["ollama-local"],
        default="ollama-local",
        help="LLM model to use (only ollama-local supported).",
    )

    args = parser.parse_args()

    if args.mode == "single":
        print_status("Initializing Single Researcher Agent...", "bold cyan")

        try:
            run_experiment_loop(args.task, test_mode=args.test_mode, model=args.model)
        except KeyboardInterrupt:
            print_status("\nExperiment interrupted by user.", "bold red")
            sys.exit(0)
        except Exception as e:
            import traceback
            print_status(f"\nFatal Error: {e}", "bold red")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        print_status("Initializing Orchestrator Agent...", "bold cyan")

        try:
            from orchestrator import run_orchestrator_loop

            run_orchestrator_loop(
                research_task=args.task,
                num_initial_agents=args.num_agents,
                max_rounds=args.max_rounds,
                default_gpu=args.gpu,
                max_parallel_experiments=args.max_parallel,
                test_mode=args.test_mode,
                model=args.model,
            )
        except KeyboardInterrupt:
            print_status("\nOrchestration interrupted by user.", "bold red")
            sys.exit(0)
        except Exception as e:
            import traceback
            print_status(f"\nFatal Error: {e}", "bold red")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
