// Route a support message with Jeb, then act, confirm or escalate. Node 22+ runs this file as is
// (node route.ts); in a browser or Deno, the same fetch works.
const JEB = "http://localhost:8100/v1/systemone"; // jeb serve; for AINode: http://<node>:3000/v1/systemone

type ChoiceAnswer = { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> };
type Action = "act" | "confirm" | "escalate";

async function route(message: string): Promise<{ action: Action; team: string; p: number }> {
  const res = await fetch(JEB, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state: { message },
      questions: {
        team: {
          type: "choice",
          instructions: "Which team should handle this message?",
          criteria: { billing: "charges, invoices, refunds", technical: "bugs, errors, outages",
                      account: "login, password, cancelling", sales: "pricing, upgrades" },
        },
      },
    }),
  });
  if (!res.ok) throw new Error(`Jeb said ${res.status}: ${await res.text()}`);
  const answer = (await res.json()).answers.team as ChoiceAnswer;
  const p = answer.probabilities[answer.choice];
  const action: Action = p >= 0.9 ? "act" : p >= 0.6 ? "confirm" : "escalate";
  return { action, team: answer.choice, p };
}

for (const msg of [
  "Hi, I was charged twice for my March subscription. Can you refund the duplicate?",
  "The export button does nothing and I can't log in on the mobile app either.",
]) {
  const r = await route(msg);
  console.log(`${r.action.padEnd(9)} ${r.team.padEnd(10)} ${r.p.toFixed(3)}  ${msg.slice(0, 60)}`);
}
