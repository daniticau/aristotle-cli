"""Rich CLI interface for the Aristotle Agent."""

import queue
import random
import shutil
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
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

_THINKING_VERBS = [
    "Contemplating",
    "Reasoning",
    "Deliberating",
    "Reflecting",
    "Considering",
    "Examining",
    "Analyzing",
    "Pondering",
    "Meditating",
    "Inquiring",
    "Discerning",
    "Syllogizing",
]

_LOADING_PHASES = [
    "Unrolling the scrolls",
    "Recalling the Categories",
    "Reviewing the Ethics",
    "Ordering first principles",
    "Warming up the Lyceum",
    "Distinguishing genus from species",
    "Preparing the dialectic",
    "Recollecting the virtues",
    "Tuning the lyre of reason",
]

ASCII_LOGO = r"""
 █████╗ ██████╗ ██╗███████╗████████╗ ██████╗ ████████╗██╗     ███████╗       ██████╗██╗     ██╗
██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗╚══██╔══╝██║     ██╔════╝      ██╔════╝██║     ██║
███████║██████╔╝██║███████╗   ██║   ██║   ██║   ██║   ██║     █████╗  █████╗██║     ██║     ██║
██╔══██║██╔══██╗██║╚════██║   ██║   ██║   ██║   ██║   ██║     ██╔══╝  ╚════╝██║     ██║     ██║
██║  ██║██║  ██║██║███████║   ██║   ╚██████╔╝   ██║   ███████╗███████╗      ╚██████╗███████╗██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝    ╚═════╝    ╚═╝   ╚══════╝╚══════╝       ╚═════╝╚══════╝╚═╝""".lstrip("\n")


# ── Display Functions ──────────────────────────────────────────────────────

def print_welcome() -> None:
    """Print styled welcome banner."""
    banner = (
        f"[aristotle]{ASCII_LOGO}[/aristotle]\n\n"
        "[info]An agent grounded in the Nicomachean Ethics[/info]\n\n"
        "Ask a question, or type [bold]/help[/bold] for commands."
    )
    console.print(Panel(banner, border_style="gold1", padding=(1, 2), expand=False))


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
        console.print()
        user_input = console.input("[user]You:[/user] ").strip()
        console.print()
        return user_input
    except (EOFError, KeyboardInterrupt):
        return "exit"


def stream_response(agent: AristotleAgent, query: str) -> None:
    """Run the query pipeline and stream the response."""
    result: dict = {}

    def _run_ask() -> None:
        result["value"] = agent.ask(query, stream=True)

    worker = threading.Thread(target=_run_ask, daemon=True)

    # Shuffle verbs so each query gets a unique sequence
    verbs = random.sample(_THINKING_VERBS, len(_THINKING_VERBS))
    verb_idx = 0
    spinner = Spinner("dots", text=f"[info] {verbs[0]}...[/info]", style="aristotle")

    with Live(spinner, console=console, transient=True, refresh_per_second=12):
        worker.start()
        while worker.is_alive():
            worker.join(timeout=0.8)
            verb_idx = (verb_idx + 1) % len(verbs)
            spinner.update(text=f"[info] {verbs[verb_idx]}...[/info]")

    response, token_gen = result["value"]

    if response.debug:
        console.print()
        print_debug_retrieval(response.debug.chunks)
        console.print()
        print_debug_prompt(response.debug.prompt)
        console.print()

    # Print label then show thinking spinner until first token arrives
    console.print("[aristotle]Aristotle:[/aristotle]")
    verbs2 = random.sample(_THINKING_VERBS, len(_THINKING_VERBS))
    verb_idx2 = 0
    spinner2 = Spinner("dots", text=f"[info] {verbs2[0]}...[/info]", style="aristotle")
    first_tokens: list[str] = []
    token_iter = iter(token_gen)
    got_first = threading.Event()

    def _wait_first_token() -> None:
        try:
            first_tokens.append(next(token_iter))
        except StopIteration:
            pass
        got_first.set()

    tok_worker = threading.Thread(target=_wait_first_token, daemon=True)
    tok_worker.start()

    with Live(spinner2, console=console, transient=True, refresh_per_second=12):
        while not got_first.is_set():
            got_first.wait(timeout=0.8)
            verb_idx2 = (verb_idx2 + 1) % len(verbs2)
            spinner2.update(text=f"[info] {verbs2[verb_idx2]}...[/info]")
    # Smooth typewriter: buffer tokens in background, drain at constant pace
    char_queue: queue.Queue[str] = queue.Queue()
    stream_done = threading.Event()

    def _fill_buffer() -> None:
        for t in first_tokens:
            for ch in t:
                char_queue.put(ch)
        for token in token_iter:
            for ch in token:
                char_queue.put(ch)
        stream_done.set()

    filler = threading.Thread(target=_fill_buffer, daemon=True)
    filler.start()

    print("\033[?25l", end="", flush=True)  # hide terminal cursor
    CHAR_DELAY = 0.014  # ~70 chars/sec — brisk but not frantic
    GLITCH_CHARS = "αβγδεζηθικλμνξπρστφχψω"
    GLITCH_LEN = 5
    terminal_width = shutil.get_terminal_size().columns
    col = 0
    word_buf: list[str] = []
    glitch_buf: list[str] = []

    def _show_glitch() -> None:
        avail = terminal_width - col
        n = min(GLITCH_LEN, avail)
        if n <= 0:
            glitch_buf.clear()
            return
        if glitch_buf:
            glitch_buf.pop(0)
            glitch_buf.append(random.choice(GLITCH_CHARS))
        while len(glitch_buf) < n:
            glitch_buf.append(random.choice(GLITCH_CHARS))
        del glitch_buf[n:]  # keep buffer in sync with display
        trail = "".join(glitch_buf)
        print(f"\033[s{trail}\033[u", end="", flush=True)

    def _clear_glitch() -> None:
        n = len(glitch_buf)
        if n <= 0:
            return
        print(f"\033[s{' ' * n}\033[u", end="", flush=True)
        glitch_buf.clear()

    def _flush_word() -> None:
        nonlocal col
        word = "".join(word_buf)
        word_buf.clear()
        if not word:
            return
        needed = (1 if col > 0 else 0) + len(word)
        if col + needed > terminal_width and col > 0:
            _clear_glitch()
            print()
            col = 0
        if col > 0:
            print(" ", end="", flush=True)
            time.sleep(CHAR_DELAY)
            col += 1
            _show_glitch()
        for c in word:
            print(c, end="", flush=True)
            time.sleep(CHAR_DELAY)
            col += 1
            _show_glitch()

    pending_newlines = 0

    while not stream_done.is_set() or not char_queue.empty():
        try:
            ch = char_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if ch == "\n":
            _flush_word()
            _clear_glitch()
            pending_newlines += 1
        elif ch == " ":
            if pending_newlines:
                # Emit at most one blank line (2 newlines max)
                for _ in range(min(pending_newlines, 1)):
                    print()
                col = 0
                pending_newlines = 0
            _flush_word()
        else:
            if pending_newlines:
                for _ in range(min(pending_newlines, 1)):
                    print()
                col = 0
                pending_newlines = 0
            word_buf.append(ch)

    _flush_word()
    _clear_glitch()
    print("\033[?25h", end="", flush=True)  # restore terminal cursor
    print()  # newline after streaming


# ── Main Loop ──────────────────────────────────────────────────────────────

def run_cli(agent: AristotleAgent) -> None:
    """Main interactive loop."""
    print_welcome()

    # Eagerly load embeddings + ChromaDB so first query is fast
    def _load() -> None:
        agent._ensure_retriever()

    worker = threading.Thread(target=_load, daemon=True)
    phases = random.sample(_LOADING_PHASES, len(_LOADING_PHASES))
    phase_idx = 0
    spinner = Spinner("dots", text=f"[info] {phases[0]}...[/info]", style="aristotle")

    with Live(spinner, console=console, transient=True, refresh_per_second=12):
        worker.start()
        while worker.is_alive():
            worker.join(timeout=2.5)
            phase_idx = (phase_idx + 1) % len(phases)
            spinner.update(text=f"[info] {phases[phase_idx]}...[/info]")

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
