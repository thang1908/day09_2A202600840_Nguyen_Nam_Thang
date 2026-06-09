from __future__ import annotations

import argparse
from pathlib import Path

from app.graph import ShoppingAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student scaffold CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        summary = assistant.run_batch(
            test_file=Path(args.test_file),
            output_dir=assistant.settings.traces_dir,
        )
        print(f"Ran {summary['total']} cases.")
        print(f"Summary: {assistant.settings.traces_dir / 'summary.json'}")
        return

    if args.question:
        result = assistant.ask(
            args.question,
            trace_file=Path(args.trace_file) if args.trace_file else None,
        )
        print(result["final_answer"])
        return

    raise SystemExit("Provide --question or --batch.")


if __name__ == "__main__":
    main()
