#!/bin/sh
# Route a support message with Jeb through a local `jeb serve` (or AINode: change the URL).
curl -s http://localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?"},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
                          "account": "login, password, cancelling", "sales": "pricing, upgrades"}}}}'

# The same question in AINode's /v1/decide shape:
curl -s http://localhost:8100/v1/decide -H 'Content-Type: application/json' -d '{
  "state": {"message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?"},
  "questions": {"team": {"question": "Which team should handle this message?",
                         "options": ["billing", "technical", "account", "sales"]}}}'
