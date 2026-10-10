from __future__ import annotations

import json
from types import SimpleNamespace

from jebadiah_decide import cli, defaults


class FakeJeb:
    backend_name = "fake"
    model = "jeb"
    temps = {}
    calibrated = False
    cap = 20
    renderer = None

    def prepare(self, progress=None):
        return {"status": "ready"}

    def decide(self, state, questions):
        if "q" in questions:
            return {
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.8, "support": 0.2},
                    }
                }
            }
        want = defaults.EXPECTED[defaults.DEFAULT_SIZE]
        return {
            "answers": {
                "route": {"choice": "billing", "probabilities": {"billing": want["billing"]}},
                "urgent": {"noul": want["urgent"]},
            },
            "latency_ms": 1,
        }


def _common_args(**overrides):
    values = {
        "backend": "ollama",
        "size": defaults.DEFAULT_SIZE,
        "url": None,
        "model": None,
        "repo": None,
        "api_key": None,
        "raw": False,
        "precision": "8bit",
        "top_n": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_human_commands_end_with_agent_line_once(monkeypatch, capsys):
    fake = FakeJeb()
    monkeypatch.setattr(cli, "build", lambda _a: fake)

    ask = _common_args(question="Where?", options="billing,support", levels=None,
                       state='{"ticket": "x"}', text=None, quiet=False)
    assert cli.cmd_ask(ask) == 0
    output = capsys.readouterr().out.rstrip()
    assert output.endswith(cli.AGENT_LINE)
    assert output.count(cli.AGENT_LINE) == 1

    class Server:
        def serve_forever(self):
            raise KeyboardInterrupt

    import jebadiah_decide.serve
    monkeypatch.setattr(jebadiah_decide.serve, "make_server", lambda *_a, **_kw: Server())
    serve = _common_args(host="127.0.0.1", port=8100, key=None, quiet=False)
    assert cli.cmd_serve(serve) == 0
    output = capsys.readouterr().out.rstrip()
    assert output.endswith(cli.AGENT_LINE)
    assert output.count(cli.AGENT_LINE) == 1

    monkeypatch.setattr(cli, "Jeb", lambda *_a, **_kw: fake)
    assert cli.cmd_doctor(_common_args()) == 0
    output = capsys.readouterr().out.rstrip()
    assert output.endswith(cli.AGENT_LINE)
    assert output.count(cli.AGENT_LINE) == 1


def test_request_json_output_stays_clean(monkeypatch, tmp_path, capsys):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": {}, "questions": {"q": {}}}))
    answer = {"answers": {"q": {"choice": "billing"}}}

    class MachineJeb:
        def decide(self, state, questions):
            return answer

    monkeypatch.setattr(cli, "build", lambda _a: MachineJeb())
    assert cli.main(["request", str(request)]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == answer
    assert cli.AGENT_LINE not in output


def test_doctor_checks_v21_reference_probabilities(monkeypatch, capsys):
    class ReleaseJeb(FakeJeb):
        renderer = object()
        calibrated = True
        temps = {"choice": 1.0467, "noul": 1.0284}
        billing = 0.565991

        def decide(self, state, questions):
            return {
                "answers": {
                    "route": {"choice": "billing", "probabilities": {"billing": self.billing}},
                    "urgent": {"noul": 0.170974},
                },
                "latency_ms": 1,
            }

    fake = ReleaseJeb()
    monkeypatch.setattr(cli, "Jeb", lambda *_a, **_kw: fake)
    assert cli.cmd_doctor(_common_args(size="9b")) == 0
    assert "known example" in capsys.readouterr().out
    fake.billing = 0.641142  # previous 9B v2, outside the v2.1 tolerance
    assert cli.cmd_doctor(_common_args(size="9b")) == 1
    assert "expected about 0.565991" in capsys.readouterr().out
