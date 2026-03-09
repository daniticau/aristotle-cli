"""Rich CLI interface for the Aristotle Agent."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from src.inference import AristotleAgent, RetrievedChunk

# ── Theme ──────────────────────────────────────────────────────────────────
custom_theme = Theme({
    "aristotle": "bold gold1",
    "citation": "cyan",
    "debug": "magenta",
    "user": "bold green",
    "info": "dim",
})

console = Console(theme=custom_theme)


# ── Display Functions ──────────────────────────────────────────────────────

def print_welcome() -> None:
    """Print styled welcome banner."""
    banner = (
        "[aristotle]Aristotle[/aristotle]\n"
        "[info]An agent grounded in the Nicomachean Ethics[/info]\n\n"
        "Ask a question, or type [bold]/help[/bold] for commands."
    )
    console.print(Panel(banner, border_style="gold1", padding=(1, 2)))


def print_debug_retrieval(chunks: list[RetrievedChunk]) -> None:
    """Display retrieved chunks as a Rich table."""
    table = Table(title="Retrieved Passages", border_style="debug")
    table.add_column("#", style="bold", width=3)
    table.add_column("Score", style="debug", width=7)
    table.add_column("Source", style="citation", width=18)
    table.add_column("Tags", width=30)
    table.add_column("Preview", ratio=1)

    for i, chunk in enumerate(chunks, 1):
        source = f"Book {chunk.book}, Ch. {chunk.chapter}"
        tags = ", ".join(chunk.tags) if chunk.tags else ""
        preview = chunk.text[:120].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            f"{chunk.score:.3f}",
            source,
            tags,
            preview,
        )

    console.print(table)


def print_debug_prompt(messages: list[dict]) -> None:
    """Display the assembled prompt (user message only, skip system)."""
    prompt_text = ""
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = msg["role"].upper()
        prompt_text += f"--- {role} ---\n{msg['content']}\n\n"

    console.print(Panel(
        prompt_text,
        title="[debug]Assembled Prompt[/debug]",
        border_style="magenta",
        expand=True,
    ))


def print_help() -> None:
    """Print available commands."""
    help_text = (
        "[bold]Commands:[/bold]\n"
        "  [bold]/debug[/bold]   — Toggle debug mode (show retrieval + prompt)\n"
        "  [bold]/help[/bold]    — Show this help message\n"
        "  [bold]exit[/bold]     — Quit the program\n"
        "  [bold]quit[/bold]     — Quit the program"
    )
    console.print(Panel(help_text, border_style="info"))


def get_user_input() -> str:
    """Get user input with styled prompt."""
    try:
        return console.input("\n[user]You:[/user] ").strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


def stream_response(agent: AristotleAgent, query: str) -> None:
    """Run the query pipeline and stream the response."""
    # Retrieve and optionally display debug info
    response, token_gen = agent.ask(query, stream=True)

    if response.debug:
        console.print()
        print_debug_retrieval(response.debug.chunks)
        console.print()
        print_debug_prompt(response.debug.prompt)
        console.print()

    # Stream response
    console.print("\n[aristotle]Aristotle:[/aristotle]")
    for token in token_gen:
        print(token, end="", flush=True)
    print()  # newline after streaming


# ── Main Loop ──────────────────────────────────────────────────────────────

def run_cli(agent: AristotleAgent) -> None:
    """Main interactive loop."""
    print_welcome()

    while True:
        query = get_user_input()

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            console.print("[info]Farewell.[/info]")
            break

        if query.lower() == "/help":
            print_help()
            continue

        if query.lower() == "/debug":
            agent.debug = not agent.debug
            state = "ON" if agent.debug else "OFF"
            console.print(f"[debug]Debug mode: {state}[/debug]")
            continue

        try:
            stream_response(agent, query)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
