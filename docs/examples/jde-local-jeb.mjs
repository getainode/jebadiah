#!/usr/bin/env node
// Run JDE's completion check with a local Jebadiah as the judge, over one of JDE's own case files.
//
// JDE (https://github.com/Titanium-Devops/jde) asks its questions through jevJudge, which posts the Jev
// wire to one endpoint. Any /v1/systemone server works: AINode, jebadiah-serve, or
// `jebadiah-decide serve` in front of Ollama, LM Studio, llama-server, vLLM or MLX.
//
//   git clone https://github.com/Titanium-Devops/jde && (cd jde && npm install && npx tsc -p tsconfig.json)
//   export TYPESAFE_API_KEY=local-jeb          # jevJudge requires one; send your server's key if it has one
//   JDE_DIR=./jde JEB_ENDPOINT=http://127.0.0.1:8100/v1/systemone JEB_MODEL=jebadiah \
//     node docs/examples/jde-local-jeb.mjs jde/cases/completion-check-tuned.json
//
// It prints one line per case (JDE's verdict against the case's label) and a total. The deadline is
// raised from JDE's shipped 750 ms, which is a hosted budget; set it to your own node's p95.

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const jdeDir = resolve(process.env.JDE_DIR ?? "./jde");
const endpoint = process.env.JEB_ENDPOINT ?? "http://127.0.0.1:8100/v1/systemone";
const model = process.env.JEB_MODEL ?? "jebadiah";
const timeoutMs = Number(process.env.JEB_TIMEOUT_MS ?? 5000);
const casesPath = process.argv[2] ?? resolve(jdeDir, "cases/completion-check-tuned.json");

const jde = await import(pathToFileURL(resolve(jdeDir, "dist/index.js")).href);
const { completionCheck, jevJudge, nullLedger, loadPolicyBook, policyFor, COMPLETION_DECISION } = jde;

const judge = jevJudge({ endpoint, model });
const shipped = policyFor(COMPLETION_DECISION, loadPolicyBook(resolve(jdeDir, "policy.json")));
const policyEntry = { ...shipped, timeout_ms: timeoutMs };

const raw = JSON.parse(await readFile(casesPath, "utf8"));
const cases = Array.isArray(raw) ? raw : raw.cases;
let graded = 0;
let right = 0;
for (const c of cases) {
  const want = c.code_expectations?.done ?? c.expected?.done;
  const out = await completionCheck(c.state, { judge, policyEntry, ledger: nullLedger() });
  const ok = want === undefined || want === "computed_by_code" ? null : out.verdict === want;
  if (ok !== null) {
    graded += 1;
    right += ok ? 1 : 0;
  }
  const mark = ok === null ? " " : ok ? "+" : "x";
  console.log(`${mark} ${c.id}: verdict ${out.verdict ?? "none"} (label ${want ?? "none"}), action ${out.action}, ` +
    `${out.latencyMs} ms${out.error ? `, error ${out.error.reason}` : ""}`);
}
console.log(`${right} of ${graded} verdicts match the labels, judge ${model} at ${endpoint}`);
