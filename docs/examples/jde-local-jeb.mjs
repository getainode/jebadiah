#!/usr/bin/env node
// Run JDE's completion check with a local Jebadiah as the judge, over one of JDE's own case files.
//
// JDE (https://github.com/Titanium-Devops/jde) judges with a local Jeb by default: its jebJudge posts the
// Jev wire to http://localhost:8100/v1/systemone, which is where `jeb serve` listens. JDE_JEB_ENDPOINT
// points it at any other /v1/systemone server (AINode, jebadiah-serve), and JDE_JEB_MODEL names the model.
//
//   git clone https://github.com/Titanium-Devops/jde && (cd jde && npm install && npx tsc -p tsconfig.json)
//   jeb serve                                   # pip install jebadiah-decide
//   JDE_DIR=./jde node docs/examples/jde-local-jeb.mjs jde/cases/completion-check-tuned.json
//
// It prints one line per case (JDE's verdict against the case's label) and a total. JDE's shipped
// deadline is 2000 ms; this run allows 5000 (JEB_TIMEOUT_MS) so a slow first load doesn't count.

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const jdeDir = resolve(process.env.JDE_DIR ?? "./jde");
const timeoutMs = Number(process.env.JEB_TIMEOUT_MS ?? 5000);
const casesPath = process.argv[2] ?? resolve(jdeDir, "cases/completion-check-tuned.json");

const jde = await import(pathToFileURL(resolve(jdeDir, "dist/index.js")).href);
const { completionCheck, jebJudge, nullLedger, loadPolicyBook, policyFor, COMPLETION_DECISION } = jde;

const judge = jebJudge();   // JDE_JEB_ENDPOINT and JDE_JEB_MODEL move it; no key needed
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
console.log(`${right} of ${graded} verdicts match the labels, judge ${judge.id}`);
