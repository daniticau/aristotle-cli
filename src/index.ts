/**
 * CLI entry point — parse args, check env, boot agent, run CLI.
 */

import "dotenv/config";
import chalk from "chalk";
import { AristotleAgent } from "./agent.js";
import { runCli } from "./ui.js";

const HELP_FLAGS = new Set(["--help", "-h"]);
const SUPPORTED_FLAGS = new Set(["--debug", ...HELP_FLAGS]);

function printUsage(): void {
  console.log("Usage: aristotle [--debug]");
  console.log("       aristotle --help");
  console.log();
  console.log("Options:");
  console.log("  --debug    Show retrieved passages and assembled prompts");
  console.log("  --help     Show this help message");
}

function requireApiKey(): void {
  if (process.env.ANTHROPIC_API_KEY?.trim()) {
    return;
  }

  console.error(chalk.red.bold("ANTHROPIC_API_KEY not set."));
  console.error(`Add it to ${chalk.bold(".env")} or set it in your shell as:`);
  console.error(`  ${chalk.bold("ANTHROPIC_API_KEY=sk-ant-...")}`);
  process.exit(1);
}

function main(): void {
  const args = process.argv.slice(2);
  const unknownArgs = args.filter((arg) => !SUPPORTED_FLAGS.has(arg));

  if (unknownArgs.length > 0) {
    console.error(chalk.red(`Unknown option: ${unknownArgs[0]}`));
    console.error();
    printUsage();
    process.exit(1);
  }

  if (args.some((arg) => HELP_FLAGS.has(arg))) {
    printUsage();
    return;
  }

  requireApiKey();

  const agent = new AristotleAgent(args.includes("--debug"));

  runCli(agent).catch((err) => {
    console.error(chalk.red(`Fatal: ${err instanceof Error ? err.message : err}`));
    process.exit(1);
  });
}

main();
