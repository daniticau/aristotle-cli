/**
 * CLI entry point — parse args, check env, boot agent, run CLI.
 */

import "dotenv/config";
import chalk from "chalk";
import { AristotleAgent } from "./agent.js";
import { runCli } from "./ui.js";

function main(): void {
  const debug = process.argv.includes("--debug");

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error(chalk.red.bold("ANTHROPIC_API_KEY not set."));
    console.error(`Set it with: ${chalk.bold("export ANTHROPIC_API_KEY=sk-ant-...")}`);
    process.exit(1);
  }

  const agent = new AristotleAgent(debug);

  runCli(agent).catch((err) => {
    console.error(chalk.red(`Fatal: ${err instanceof Error ? err.message : err}`));
    process.exit(1);
  });
}

main();
