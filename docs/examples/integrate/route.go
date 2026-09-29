// Route a support message with Jeb, then act, confirm or escalate. go run route.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

const jeb = "http://localhost:8100/v1/systemone" // jeb serve; for AINode: http://<node>:3000/v1/systemone

type choiceAnswer struct {
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
}

func route(message string) (action, team string, p float64, err error) {
	// Options are shown to the model in the order you send them, and a Go map would sort them,
	// so the criteria stay a JSON literal in a fixed order.
	criteria := json.RawMessage(`{"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
		"account": "login, password, cancelling", "sales": "pricing, upgrades"}`)
	body, _ := json.Marshal(map[string]any{
		"state": map[string]string{"message": message},
		"questions": map[string]any{"team": map[string]any{
			"type":         "choice",
			"instructions": "Which team should handle this message?",
			"criteria":     criteria,
		}},
	})
	res, err := http.Post(jeb, "application/json", bytes.NewReader(body))
	if err != nil {
		return
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		err = fmt.Errorf("jeb said %s", res.Status)
		return
	}
	var out struct {
		Answers map[string]choiceAnswer `json:"answers"`
	}
	if err = json.NewDecoder(res.Body).Decode(&out); err != nil {
		return
	}
	a := out.Answers["team"]
	team, p = a.Choice, a.Probabilities[a.Choice]
	switch {
	case p >= 0.9:
		action = "act"
	case p >= 0.6:
		action = "confirm"
	default:
		action = "escalate"
	}
	return
}

func main() {
	for _, msg := range []string{
		"Hi, I was charged twice for my March subscription. Can you refund the duplicate?",
		"The export button does nothing and I can't log in on the mobile app either.",
	} {
		action, team, p, err := route(msg)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%-9s %-10s %.3f  %.60s\n", action, team, p, msg)
	}
}
