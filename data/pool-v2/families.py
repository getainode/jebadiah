"""State families for the Jebadiah v2 synthetic pool, and the random seed attributes that keep
the author from collapsing onto one template.

Each family says what a state is (and the JSON fields it should roughly carry) and which typed
questions make sense about it. The author writes 5 questions per state from the menu or in
the same spirit; the judge stage decides the answers, the author never writes one.

The five JDE help-desk judgment shapes (echo, parts, coverage, added requirements, parrot) are
defined here from their names only: the reference page named in the brief was reachable on
2026-09-22 but does not define them, and no example text from it is used.
"""
from __future__ import annotations

import random

FAMILIES: dict[str, dict] = {
    "helpdesk_ticket": {
        "state": "one synthetic MSP (managed service provider) help-desk ticket as a client would submit it, plus the "
                 "context a dispatcher sees: ticket id, client, requester, submitted_at timestamp, the client's SLA "
                 "(first response and resolution targets), current time, subject, body, and a short list of 2 to 5 "
                 "OTHER open tickets for the same client (id plus one-line description) so duplicates can be judged",
        "questions": [
            "routing: which queue or team should take it (choice, 4 to 8 named queues)",
            "priority: which priority level fits (choice, P1 to P4 each with a one-line definition)",
            "needs a human: can this be closed by automation or a canned answer, or does a technician have to act (noul)",
            "duplicate: is this the same underlying problem as one of the listed open tickets, and which one (choice with each listed ticket as an option plus 'none of these')",
            "breach risk: given the timestamps and the SLA, will the first-response target be missed if nobody acts in the next hour (noul)",
        ],
        "edges": ["the SLA clock is almost out", "the listed open ticket describes the same fault in other words",
                  "two issues in one ticket", "the requester says urgent but the impact is small",
                  "an automated alert forwarded by a person", "the timestamps are in two different time zones"],
    },
    "cs_chat": {
        "state": "a short customer-service chat transcript (3 to 9 turns between a customer and an agent or bot) for a "
                 "consumer business, with channel, account tier and the policy notes the agent has, ending on the "
                 "customer's latest message",
        "questions": [
            "intent of the latest customer turn (choice, 5 to 12 intents)",
            "next best action for the agent (choice, 3 to 8 actions)",
            "escalate to a human supervisor now (noul)",
            "is the customer asking for something the stated policy forbids (noul)",
        ],
        "edges": ["the customer changes topic mid-chat", "sarcasm", "the bot already gave a wrong answer",
                  "a refund request just outside the policy window", "legal threat", "the customer is polite but churning"],
    },
    "agent_tool_selection": {
        "state": "a trace from an AI agent: the user's request, the list of available tools (name plus one-line "
                 "description, 4 to 10 tools), and the steps already taken with their tool outputs",
        "questions": [
            "which tool should the agent call next (choice over the listed tool names plus 'answer the user now')",
            "is the information needed to answer already present in the trace (noul)",
            "does the next call need information the user has not given (noul)",
            "which argument of the planned call is wrong or missing (choice)",
        ],
        "edges": ["a tool returned an error", "two tools look similar", "the user's request is ambiguous",
                  "the last tool output already answers it", "a destructive tool is available"],
    },
    "invoice_receipt": {
        "state": "the text of an invoice or receipt as extracted by OCR (vendor, dates, line items, taxes, totals, "
                 "payment terms, remit-to), possibly with OCR noise, plus the purchase order or expense policy it is checked against",
        "questions": [
            "which field is inconsistent or wrong (choice over field names plus 'none')",
            "expense category (choice, 5 to 10 categories)",
            "do the line items add up to the stated total (noul)",
            "is it payable under the stated policy or PO (noul)",
            "which approval step it needs next (choice)",
        ],
        "edges": ["the tax is computed on the wrong base", "the currency differs from the PO", "a duplicate line item",
                  "a handwritten tip", "the due date is before the invoice date", "OCR swapped 1 and 7"],
    },
    "meeting_actions": {
        "state": "meeting notes or a transcript excerpt (attendees with roles, date, discussion) in which action items "
                 "are assigned more or less explicitly",
        "questions": [
            "who owns a named action item (choice over attendees plus 'unassigned')",
            "what due date was agreed for it (choice over candidate dates plus 'no date set')",
            "was a decision actually made on a named topic, or only discussed (noul)",
            "is a named follow-up blocked on someone outside the meeting (noul)",
        ],
        "edges": ["an owner volunteers then hands it off", "a relative date like 'end of next week'",
                  "two people share a first name", "the decision is reversed later in the notes", "notes in bullet fragments"],
    },
    "incident_alert": {
        "state": "a monitoring alert or incident page (source system, metric, threshold, current value, duration, "
                 "affected service, recent deploys, related alerts, runbook excerpt)",
        "questions": [
            "severity (choice SEV1 to SEV4 each with a one-line definition)",
            "is the alert actionable or noise (noul)",
            "most likely cause among listed hypotheses (choice)",
            "should the on-call be paged right now (noul)",
            "which team owns it (choice)",
        ],
        "edges": ["a flapping alert", "a maintenance window is in effect", "a deploy 5 minutes before",
                  "the metric is bad but no user impact", "a cascade from an upstream dependency"],
    },
    "email_routing": {
        "state": "one inbound email to a shared inbox (from, to, cc, subject, date, body, attachments listed by name) "
                 "plus the list of teams or mailboxes it could be routed to with one-line remits",
        "questions": [
            "which team should get it (choice over the listed teams)",
            "is it spam, phishing or legitimate (choice)",
            "does it need a reply within one business day (noul)",
            "is it a reply to an existing thread rather than a new request (noul)",
        ],
        "edges": ["a phishing email that looks like a vendor invoice", "a legitimate email from a free-mail domain",
                  "the request fits two teams", "an auto-reply", "a forwarded chain where the real ask is at the bottom"],
    },
    "content_policy": {
        "state": "a user-generated post or comment on a platform, its context (surface, thread, reporter note) and a "
                 "short numbered content policy of 4 to 8 rules. Keep it realistic but never graphic or hateful in the text itself; describe severe material abstractly",
        "questions": [
            "which rule applies, if any (choice over the rules plus 'no rule violated')",
            "does it need human review before any action (noul)",
            "what action fits (choice: leave up, add a warning notice, reduce reach, remove, escalate)",
            "is it satire or quotation rather than an endorsement (noul)",
        ],
        "edges": ["quoting a slur to criticize it", "medical misinformation phrased as a question",
                  "a borderline promotion", "self-harm mentioned in the past tense", "a joke between friends"],
    },
    "product_review": {
        "state": "one product review (product, category, star rating given, title, text, verified purchase) for a "
                 "physical product, app or service",
        "questions": [
            "which aspect is the main complaint or praise (choice, 4 to 8 aspects)",
            "sentiment about a named aspect (choice: positive, negative, mixed, not mentioned)",
            "does the text mention a defect that needs a safety review (noul)",
            "is the star rating consistent with the text (noul)",
            "would the reviewer buy again (choice: yes, no, not stated)",
        ],
        "edges": ["5 stars with a complaint", "a review about shipping, not the product", "mixed sentiment across aspects",
                  "a review that compares to a competitor", "non-native English"],
    },
    "log_severity": {
        "state": "a short excerpt of application or system log lines (3 to 15 lines, realistic formats: syslog, JSON "
                 "logs, stack traces, access logs) with the service name and the time window",
        "questions": [
            "severity of the excerpt as a whole (choice: debug, info, warning, error, critical)",
            "which component is failing (choice)",
            "does it indicate data loss or corruption (noul)",
            "is this a known-benign pattern that can be suppressed (noul)",
            "which line is the root event (choice over line numbers)",
        ],
        "edges": ["an ERROR-level line that is actually harmless", "a WARN that precedes an outage",
                  "a retry loop that eventually succeeds", "interleaved lines from two requests", "a truncated stack trace"],
    },
    "code_review_comment": {
        "state": "one code review comment in context: the language, a small diff hunk (5 to 25 lines), the "
                 "reviewer's comment, and optionally the author's reply",
        "questions": [
            "is the comment blocking (must fix before merge) or not (noul)",
            "category of the comment (choice: correctness, security, performance, readability, style, testing, docs, design)",
            "is the reviewer's claim about the code correct (noul)",
            "what should the author do (choice: fix, reply and keep, ask for clarification, defer to a follow-up)",
        ],
        "edges": ["a nit phrased as a demand", "a real bug phrased as a question", "the reviewer is wrong",
                  "a security issue hidden in a style comment", "the author's reply already fixed it"],
    },
    "pr_triage": {
        "state": "a pull request as a maintainer sees it: title, description, author (first-time or regular), files "
                 "changed with line counts, CI status, linked issue, tags already applied, and 0 to 3 comments. Use the key 'tags', never 'labels'",
        "questions": [
            "which area owner should review (choice)",
            "type of change (choice: bug fix, feature, refactor, docs, dependency bump, test, chore)",
            "can it be merged once approved, or does it need more work first (noul)",
            "does it need a changelog entry (noul)",
            "risk level (choice: low, medium, high each with a one-line definition)",
        ],
        "edges": ["CI red for an unrelated flaky test", "a huge PR described as a small fix",
                  "a dependency bump with a breaking major version", "a docs PR that changes code", "no linked issue"],
    },
    "scheduling_dispatch": {
        "state": "a dispatch or scheduling situation: a job or appointment request (location, window, skills needed, "
                 "duration) and a roster of 3 to 8 technicians or rooms with availability, skills, location and current load",
        "questions": [
            "who or what should be assigned (choice over the roster)",
            "can the request be met within its window (noul)",
            "which constraint is the binding one (choice)",
            "does it need the customer to be contacted to reschedule (noul)",
        ],
        "edges": ["only one qualified tech and they are overbooked", "a time-zone mismatch", "travel time makes the window impossible",
                  "a double-booked room", "an urgent job that preempts a routine one"],
    },
    "sales_lead": {
        "state": "an inbound sales lead: form fields (company, size, role, budget band, timeline, use case, how they "
                 "heard), enrichment data, and the qualification criteria the sales team uses",
        "questions": [
            "qualification stage (choice: disqualify, nurture, marketing qualified, sales qualified)",
            "which product tier fits (choice)",
            "is the contact a decision maker (noul)",
            "does the lead meet the stated budget threshold (noul)",
            "which rep or segment should own it (choice)",
        ],
        "edges": ["a student using a work email", "a competitor researching", "budget not stated but company is large",
                  "an existing customer asking for an upsell", "the timeline is 'just looking'"],
    },
    "form_validation": {
        "state": "a filled-in form submission (8 to 20 fields: names, addresses, dates, IDs, amounts, checkboxes) with "
                 "the form's validation rules written out",
        "questions": [
            "which field is wrong (choice over field names plus 'all valid')",
            "is the submission complete enough to process (noul)",
            "why is the named field invalid (choice over reasons)",
            "does it need manual review for possible fraud (noul)",
        ],
        "edges": ["a date of birth that makes the applicant a minor", "a postcode that does not match the city",
                  "two fields that contradict each other", "a valid but unusual name", "an IBAN with a bad checksum"],
    },
    "claim_evidence": {
        "state": "a claim and a piece of evidence text (a paragraph from a report, article, dataset description or "
                 "email) on any everyday or professional topic, written fresh",
        "questions": [
            "is the claim supported by the evidence (noul)",
            "relation (choice: supported, contradicted, not enough information)",
            "which sentence of the evidence is decisive (choice over numbered sentences plus 'none')",
            "does the claim overstate the evidence (noul)",
        ],
        "edges": ["numbers that almost match", "a claim about causation when the evidence is correlation",
                  "the evidence is about a different year", "a hedged claim", "the evidence supports a part of the claim"],
    },
    "summary_faithfulness": {
        "state": "a source text (an internal memo, meeting recap, product announcement, support thread or short article, "
                 "written fresh) and a summary of it",
        "questions": [
            "does the summary state something the source does not (noul)",
            "which kind of error does the summary make (choice: none, invented fact, wrong number, wrong entity, wrong causality, omitted key point)",
            "does the summary leave out the main point (noul)",
            "which summary sentence is unsupported (choice over numbered sentences plus 'none')",
        ],
        "edges": ["the summary merges two people", "a rounded number", "the summary is faithful but vague",
                  "a negation dropped", "the summary adds a plausible but absent reason"],
    },
    "translation_adequacy": {
        "state": "a source sentence or short passage in one language and a candidate translation into another, with "
                 "the domain (legal, medical, UI string, marketing, casual chat, technical)",
        "questions": [
            "is the translation adequate, conveying the full meaning (noul)",
            "main error type (choice: none, mistranslation, omission, addition, wrong register, untranslated term, number or unit error)",
            "is the register right for the domain (noul)",
            "severity of the worst error (choice: none, minor, major, critical)",
        ],
        "edges": ["a dropped negation", "a false friend", "a unit converted wrongly", "too formal for a chat",
                  "a placeholder variable broken in a UI string", "a perfect translation"],
    },
    "voice_intent": {
        "state": "one voice-assistant turn as recognized by speech-to-text (possibly with recognition errors), plus "
                 "the device, the time, and the previous turn if any. Write your own utterance; do not reuse any "
                 "benchmark's phrasing",
        "questions": [
            "intent (choice, 5 to 12 intents for a home or car assistant)",
            "which slot value is the target (choice)",
            "does the assistant need to ask a follow-up question before acting (noul)",
            "is the turn addressed to the assistant at all (noul)",
        ],
        "edges": ["a speech recognition error changes a word", "a follow-up that depends on the previous turn",
                  "two requests in one utterance", "a request the device cannot do", "background speech captured"],
    },
    "jde_echo": {
        "state": "a help-desk ticket from a client and the technician's draft reply to it",
        "questions": [
            "echo: does the reply only restate the client's problem back to them without adding a diagnosis, a step or a question that moves it forward (noul)",
            "what does the reply add beyond restating (choice: nothing, a diagnosis, a next step, a clarifying question, a resolution)",
            "does the reply show the technician understood the request correctly (noul)",
        ],
        "edges": ["a long reply that says nothing new", "a short reply that fixes it", "a reply that restates then asks one good question",
                  "a reply to a different issue than the one asked"],
    },
    "jde_parts": {
        "state": "a help-desk ticket whose request has several distinct parts (2 to 6 asks), and the technician's reply or closing note",
        "questions": [
            "parts: does the reply address every part of the request (noul)",
            "which part is left unaddressed (choice over the parts, named briefly, plus 'none')",
            "how many of the parts are addressed (choice over counts)",
        ],
        "edges": ["the unaddressed part is buried in a P.S.", "one part is addressed only partly",
                  "the reply addresses a part nobody asked about", "all parts addressed in one sentence each"],
    },
    "jde_coverage": {
        "state": "a help-desk ticket and the resolution a technician recorded (what they did, what they changed), possibly with the client's confirmation",
        "questions": [
            "coverage: does the recorded resolution cover the problem as reported (choice: fully, partly, not at all)",
            "is the ticket safe to close now (noul)",
            "which reported symptom is not explained by the resolution (choice over symptoms plus 'none')",
        ],
        "edges": ["the fix covers one of two affected users", "a workaround recorded as a fix",
                  "the client confirms but mentions a new symptom", "the fix is for a different device"],
    },
    "jde_added_requirements": {
        "state": "a help-desk request from a client and the plan, quote or reply the technician wrote in response",
        "questions": [
            "added requirements: does the response add requirements the client never asked for (noul)",
            "which added item is not in the request (choice over items in the response plus 'none')",
            "is the added work necessary to meet the request (noul)",
        ],
        "edges": ["a necessary prerequisite the client did not know about", "an upsell framed as required",
                  "a scope that silently shrinks", "the response matches the request exactly"],
    },
    "jde_parrot": {
        "state": "a help-desk ticket in which the client states a cause or fact (possibly wrong), and the technician's note or reply",
        "questions": [
            "parrot: does the technician repeat the client's claimed cause as fact without checking it (noul)",
            "what did the technician do with the client's claim (choice: verified it, disproved it, repeated it unchecked, ignored it)",
            "is the client's stated cause supported by the evidence in the ticket (noul)",
        ],
        "edges": ["the client blames a recent update that is unrelated", "the technician checks and confirms",
                  "the technician copies the client's wording into the resolution", "the claim is plausible but untested"],
    },
}

INDUSTRIES = [
    "dental clinic", "law firm", "accounting practice", "regional hospital", "veterinary clinic", "architecture studio",
    "construction contractor", "logistics warehouse", "e-commerce shop", "SaaS startup", "public library", "school district",
    "credit union", "insurance broker", "real estate agency", "restaurant group", "hotel", "manufacturing plant",
    "nonprofit charity", "municipal government", "university research lab", "retail chain", "trucking company",
    "marketing agency", "biotech startup", "home health agency", "car dealership", "gym franchise", "airline ground ops",
    "utility company", "game studio", "newsroom", "church", "farm cooperative", "museum", "telecom reseller",
    "pharmacy", "property management firm", "film production company", "solar installer", "freight forwarder",
    "mobile app developer", "consumer electronics brand", "online marketplace", "travel agency",
]
PERSONAS = [
    "an office manager", "a junior employee", "a senior executive", "a non-technical owner", "an engineer",
    "a contractor", "a customer in a hurry", "an elderly customer", "a non-native English speaker", "a power user",
    "an intern", "an automated system", "a finance clerk", "a nurse", "a teacher", "a field technician",
    "a project manager", "a reseller partner", "a security analyst", "a first-time user",
]
TONES = ["neutral", "frustrated", "polite and apologetic", "terse", "rambling", "angry", "formal", "casual with typos",
         "anxious", "sarcastic", "matter-of-fact", "confused"]
LENGTHS = {"short": "short (about 40 to 100 words of state text)",
           "medium": "medium (about 100 to 250 words of state text)",
           "long": "long (about 250 to 450 words of state text)"}
GENERIC_EDGES = ["key information is missing", "two details conflict", "it contains irrelevant detail",
                 "it is a borderline case between two options", "it is a clear, easy case", "it is a clear, easy case",
                 "names and numbers must be read carefully", "it mixes two unrelated issues"]
LANG_PAIRS = ["English to German", "German to English", "English to Spanish", "Spanish to English", "English to French",
              "French to English", "English to Portuguese", "Italian to English", "English to Dutch", "Japanese to English",
              "English to Japanese", "Polish to English", "English to Swedish", "Chinese to English"]


def seed_attrs(family: str, rng: random.Random) -> dict:
    fam = FAMILIES[family]
    a = {
        "industry": rng.choice(INDUSTRIES),
        "persona": rng.choice(PERSONAS),
        "tone": rng.choice(TONES),
        "length": rng.choices(list(LENGTHS), weights=[4, 5, 1])[0],
        "edge": rng.choice(fam["edges"] + GENERIC_EDGES),
        "n_questions": 5,
        "year": rng.choice([2025, 2026, 2026, 2027]),
        "month": rng.randint(1, 12),
    }
    if family == "translation_adequacy":
        a["lang_pair"] = rng.choice(LANG_PAIRS)
    return a
