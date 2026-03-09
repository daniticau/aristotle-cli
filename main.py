"""Entry point for the Aristotle Agent CLI."""

import argparse
import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Aristotle Agent — ask questions grounded in the Nicomachean Ethics")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (show retrieval and prompt)")
    args = parser.parse_args()

    # Check dependencies
    try:
        from rich.console import Console
    except ImportError:
        print("Missing dependency: rich. Install with: pip install rich")
        sys.exit(1)

    console = Console()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]ANTHROPIC_API_KEY not set.[/bold red]")
        console.print("Set it with: [bold]set ANTHROPIC_API_KEY=sk-...[/bold]")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        console.print("[bold red]Missing dependency:[/bold red] anthropic. Install with: pip install anthropic")
        sys.exit(1)

    try:
        import chromadb
    except ImportError:
        console.print("[bold red]Missing dependency:[/bold red] chromadb. Install with: pip install chromadb")
        sys.exit(1)

    # Load agent with spinner
    from src.inference import AristotleAgent
    from src.ui import run_cli

    try:
        with console.status("[bold gold1]Loading Aristotle's mind...[/bold gold1]", spinner="dots"):
            agent = AristotleAgent(debug=args.debug)
    except RuntimeError as e:
        console.print(f"[bold red]Setup error:[/bold red] {e}")
        sys.exit(1)

    # Run interactive loop
    try:
        run_cli(agent)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Farewell.[/dim]")


if __name__ == "__main__":
    main()
