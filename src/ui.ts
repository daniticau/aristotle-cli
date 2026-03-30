/**
 * CLI interface — banner, spinners, streaming output, input loop.
 */

import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import chalk from "chalk";
import ora from "ora";
import Table from "cli-table3";
import { AristotleAgent, type DebugInfo } from "./agent.js";
import type { RetrievedChunk } from "./retriever.js";

// ── Constants ────────────────────────────────────────────────────────────

const gold = chalk.hex("#FFD700");
const goldBold = chalk.hex("#FFD700").bold;
const dim = chalk.dim;
const green = chalk.green.bold;
const magenta = chalk.magenta;

const THINKING_VERBS = [
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
];

const LOADING_PHASES = [
  "Unrolling the scrolls",
  "Recalling the Categories",
  "Reviewing the Ethics",
  "Ordering first principles",
  "Warming up the Lyceum",
  "Distinguishing genus from species",
  "Preparing the dialectic",
  "Recollecting the virtues",
  "Tuning the lyre of reason",
  "Consulting the Metaphysics",
  "Reviewing the Politics",
  "Studying the Rhetoric",
  "Contemplating the Physics",
];

const ASCII_LOGO = `
 ${goldBold("█████╗ ██████╗ ██╗███████╗████████╗ ██████╗ ████████╗██╗     ███████╗       ██████╗██╗     ██╗")}
${goldBold("██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗╚══██╔══╝██║     ██╔════╝      ██╔════╝██║     ██║")}
${goldBold("███████║██████╔╝██║███████╗   ██║   ██║   ██║   ██║   ██║     █████╗  █████╗██║     ██║     ██║")}
${goldBold("██╔══██║██╔══██╗██║╚════██║   ██║   ██║   ██║   ██║   ██║     ██╔══╝  ╚════╝██║     ██║     ██║")}
${goldBold("██║  ██║██║  ██║██║███████║   ██║   ╚██████╔╝   ██║   ███████╗███████╗      ╚██████╗███████╗██║")}
${goldBold("╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝    ╚═════╝    ╚═╝   ╚══════╝╚══════╝       ╚═════╝╚══════╝╚═╝")}
`;

const GLITCH_CHARS = "αβγδεζηθικλμνξπρστφχψω";
const GLITCH_LEN = 5;
const CHAR_DELAY = 8; // ms — ~125 chars/sec

// ── Helpers ──────────────────────────────────────────────────────────────

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomGlitchChar(): string {
  return GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)];
}

function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*m/g, "");
}

// ── Display Functions ────────────────────────────────────────────────────

function printWelcome(): void {
  const contentLines = [
    ...ASCII_LOGO.split("\n").filter(Boolean),
    "",
    dim("  An agent grounded in the complete works of Aristotle"),
    "",
    `  Ask a question, or type ${chalk.bold("/help")} for commands.`,
  ];

  const hPad = 2;
  const vPad = 1;
  const maxWidth = Math.max(...contentLines.map((l) => stripAnsi(l).length));
  const innerWidth = maxWidth + hPad * 2;

  const emptyRow = gold("│") + " ".repeat(innerWidth) + gold("│");

  console.log();
  console.log(gold(`╭${"─".repeat(innerWidth)}╮`));
  for (let i = 0; i < vPad; i++) console.log(emptyRow);
  for (const line of contentLines) {
    const visualLen = stripAnsi(line).length;
    const rightPad = maxWidth - visualLen + hPad;
    console.log(
      gold("│") + " ".repeat(hPad) + line + " ".repeat(rightPad) + gold("│"),
    );
  }
  for (let i = 0; i < vPad; i++) console.log(emptyRow);
  console.log(gold(`╰${"─".repeat(innerWidth)}╯`));
  console.log();
}

function printDebugRetrieval(chunks: RetrievedChunk[]): void {
  const table = new Table({
    head: [
      magenta("#"),
      magenta("Score"),
      chalk.cyan("Source"),
      "Tags",
      "Preview",
    ],
    colWidths: [4, 9, 20, 32, 50],
    wordWrap: true,
  });

  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const source = `${c.work}\nBk ${c.book}, Pt ${c.chapter}`;
    const tags = c.tags.join(", ");
    const preview = c.text.slice(0, 120).replace(/\n/g, " ") + "...";
    table.push([
      String(i + 1),
      magenta(c.score.toFixed(3)),
      chalk.cyan(source),
      tags,
      preview,
    ]);
  }

  console.log();
  console.log(magenta("  Retrieved Passages"));
  console.log(table.toString());
}

function printDebugPrompt(
  prompt: Array<{ role: string; content: string }>,
): void {
  console.log();
  console.log(magenta("  Assembled Prompt"));
  for (const msg of prompt) {
    if (msg.role === "system") continue;
    console.log(magenta(`  --- ${msg.role.toUpperCase()} ---`));
    console.log(msg.content);
  }
  console.log();
}

function printHelp(): void {
  console.log();
  console.log(chalk.bold("  Commands:"));
  console.log(`  ${chalk.bold("/debug")}   — Toggle debug mode (show retrieval + prompt)`);
  console.log(`  ${chalk.bold("/help")}    — Show this help message`);
  console.log(`  ${chalk.bold("exit")}     — Quit the program`);
  console.log(`  ${chalk.bold("quit")}     — Quit the program`);
  console.log();
}

// ── Spinner with cycling text ────────────────────────────────────────────

function cyclingSpinner(
  phrases: string[],
  intervalMs: number,
): { spinner: ReturnType<typeof ora>; stop: () => void } {
  const shuffled = shuffle(phrases);
  let idx = 0;
  const spinner = ora({
    text: dim(` ${shuffled[0]}...`),
    color: "yellow",
  }).start();

  const timer = setInterval(() => {
    idx = (idx + 1) % shuffled.length;
    spinner.text = dim(` ${shuffled[idx]}...`);
  }, intervalMs);

  return {
    spinner,
    stop: () => {
      clearInterval(timer);
      spinner.stop();
    },
  };
}

// ── Greek glitch-trail typewriter ────────────────────────────────────────

class GlitchTypewriter {
  private col = 0;
  private terminalWidth: number;
  private wordBuf: string[] = [];
  private glitchBuf: string[] = [];

  constructor() {
    this.terminalWidth = stdout.columns || 80;
  }

  private showGlitch(): void {
    const avail = this.terminalWidth - this.col;
    const n = Math.min(GLITCH_LEN, avail);
    if (n <= 0) {
      this.glitchBuf.length = 0;
      return;
    }
    if (this.glitchBuf.length > 0) {
      this.glitchBuf.shift();
      this.glitchBuf.push(randomGlitchChar());
    }
    while (this.glitchBuf.length < n) {
      this.glitchBuf.push(randomGlitchChar());
    }
    this.glitchBuf.length = n;
    const trail = this.glitchBuf.join("");
    stdout.write(`\x1b[s${trail}\x1b[u`);
  }

  private clearGlitch(): void {
    const n = this.glitchBuf.length;
    if (n <= 0) return;
    stdout.write("\x1b[K");
    this.glitchBuf.length = 0;
  }

  private async flushWord(): Promise<void> {
    const word = this.wordBuf.join("");
    this.wordBuf.length = 0;
    if (!word) return;

    const needed = (this.col > 0 ? 1 : 0) + word.length;
    if (this.col + needed > this.terminalWidth && this.col > 0) {
      this.clearGlitch();
      stdout.write("\n");
      this.col = 0;
    }
    if (this.col > 0) {
      stdout.write(" ");
      await sleep(CHAR_DELAY);
      this.col++;
      this.showGlitch();
    }
    for (const c of word) {
      stdout.write(c);
      await sleep(CHAR_DELAY);
      this.col++;
      this.showGlitch();
    }
  }

  async write(tokenGen: AsyncGenerator<string>): Promise<void> {
    // Hide cursor during typewriter
    stdout.write("\x1b[?25l");

    let pendingNewlines = 0;

    for await (const token of tokenGen) {
      for (const ch of token) {
        if (ch === "\n") {
          await this.flushWord();
          this.clearGlitch();
          pendingNewlines++;
        } else if (ch === " ") {
          if (pendingNewlines > 0) {
            for (let i = 0; i < Math.min(pendingNewlines, 1); i++) {
              stdout.write("\n");
            }
            this.col = 0;
            pendingNewlines = 0;
          }
          await this.flushWord();
        } else {
          if (pendingNewlines > 0) {
            for (let i = 0; i < Math.min(pendingNewlines, 1); i++) {
              stdout.write("\n");
            }
            this.col = 0;
            pendingNewlines = 0;
          }
          this.wordBuf.push(ch);
        }
      }
    }

    await this.flushWord();
    this.clearGlitch();
    // Restore cursor and add scroll padding
    stdout.write("\x1b[?25h");
    stdout.write("\n");
  }
}

// ── Stream response ──────────────────────────────────────────────────────

async function streamResponse(
  agent: AristotleAgent,
  query: string,
): Promise<void> {
  // Show thinking spinner while retrieving + building prompt
  const thinking = cyclingSpinner(THINKING_VERBS, 800);

  let response: Awaited<ReturnType<typeof agent.ask>>;
  try {
    response = await agent.ask(query);
  } finally {
    thinking.stop();
  }

  const [agentResponse, tokenGen] = response;

  if (agentResponse.debug) {
    printDebugRetrieval(agentResponse.debug.chunks);
    printDebugPrompt(agentResponse.debug.prompt);
  }

  // Print label
  stdout.write(goldBold("Aristotle: ") + "\n");

  // Wait for first token with spinner, then stop spinner before typewriter starts
  const firstTokenSpinner = cyclingSpinner(THINKING_VERBS, 800);
  const firstResult = await tokenGen.next();
  firstTokenSpinner.stop();

  if (firstResult.done) return;

  async function* prependFirst(): AsyncGenerator<string> {
    yield firstResult.value;
    yield* tokenGen;
  }

  const typewriter = new GlitchTypewriter();
  await typewriter.write(prependFirst());
}

// ── Main CLI loop ────────────────────────────────────────────────────────

export async function runCli(agent: AristotleAgent): Promise<void> {
  printWelcome();

  // Eagerly load embeddings + download model with spinning status
  const loading = cyclingSpinner(LOADING_PHASES, 2500);
  try {
    await agent.ensureRetriever();
  } finally {
    loading.stop();
  }

  while (true) {
    console.log();

    const rl = createInterface({ input: stdin, output: stdout });
    let query: string;
    try {
      query = (await rl.question(green("You: "))).trim();
    } catch {
      rl.close();
      break;
    }
    rl.close();

    console.log();

    if (!query) continue;

    const lower = query.toLowerCase();
    if (lower === "exit" || lower === "quit") {
      console.log(dim("Farewell."));
      break;
    }

    if (lower === "/help") {
      printHelp();
      continue;
    }

    if (lower === "/debug") {
      agent.debug = !agent.debug;
      const state = agent.debug ? "ON" : "OFF";
      console.log(magenta(`Debug mode: ${state}`));
      continue;
    }

    try {
      await streamResponse(agent, query);
    } catch (err) {
      console.error(chalk.red(`Error: ${err instanceof Error ? err.message : err}`));
    }
  }
}
