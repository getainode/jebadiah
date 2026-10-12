"""CPU checks for opt-in Decision Index policies, without model weights."""
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'train'))
import jebadiah_prompt as jp
from inference_renderer import repeat_renderer


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Only the kit's base class is needed for these weight-free behavioral tests.
kit = ModuleType('decision_index.engines')
kit.Engine = type('Engine', (), {})
kit.Unsupported = type('Unsupported', (Exception,), {})
with patch.dict(sys.modules, {'decision_index.engines': kit}):
    engine_module = load('variant_engine', Path(__file__).with_name('jebadiah_engine.py'))
legacy = load('legacy_call', Path(__file__).with_name('fixtures') / 'legacy_call.py')
legacy.Unsupported = kit.Unsupported


class Tokenizer:
    """Deterministic token ids; labels stay single tokens at the answer boundary."""
    all_special_ids = []
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8')

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages) + '<assistant>'


class Scorer:
    def __init__(self, repeat=False):
        self.tok = Tokenizer()
        self.renderer = repeat_renderer(jp, self.tok, 100000, repeat_payload=repeat)
        self.seen = []

    def render(self, state, question):
        return self.renderer.render(state, question)

    def score_rendered(self, items):
        self.seen.extend(items)
        out = []
        for r, _ in items:
            # Both semantic preference and positional bias, to catch incorrect key mapping.
            weights = [(i + 1) * (2 if k == 'best' else 1) for i, k in enumerate(r.keys)]
            out.append([w / sum(weights) for w in weights])
        return out


class OOM(Exception):
    pass


def engine(reverse=False, repeat=False, prefix=False):
    e = engine_module.JebadiahEngine.__new__(engine_module.JebadiahEngine)
    e.reverse_avg, e.repeat_payload, e.prefix_cache = reverse, repeat, prefix
    e.scorer = Scorer(repeat)
    e.tok = e.scorer.tok
    e.answer_from_probs = jp.answer_from_probs
    e.flatten_text = jp.flatten_text
    e.context = 100000
    e.token_budget, e.max_batch, e.batch_tokens = 32768, 8, None
    e.trace, e.oom_retries, e.device = None, 0, 'cpu'
    e.torch = SimpleNamespace(OutOfMemoryError=OOM)
    e.pad_waste, e.prefix_align = .15, 64
    e.model_id = 'toy'
    return e


CHOICE = {'type': 'choice', 'instructions': 'Pick one', 'criteria': {'first': 'One', 'best': 'Two', 'last': 'Three'}}
NOUL = {'type': 'noul', 'instructions': 'Is this true?', 'criteria': None}


class VariantsTest(unittest.TestCase):
    def test_defaults_match_frozen_engine_prompts_tokens_and_outputs(self):
        questions = {'c': CHOICE, 'n': NOUL, 'translated': {**CHOICE, 'instructions': {'q': 'Pick', 'focus': 'carefully'}}}
        old, new = engine(), engine()
        expected = legacy.__call__(old, {'facts': 'payload'}, questions)
        actual = new({'facts': 'payload'}, questions)
        self.assertEqual(actual, expected)
        for (r1, t1), (r2, t2) in zip(old.scorer.seen, new.scorer.seen, strict=True):
            self.assertEqual(r1.prompt.encode(), r2.prompt.encode())
            self.assertEqual(old.tok.encode(r1.prompt), new.tok.encode(r2.prompt))
            self.assertEqual((r1.cand_ids, t1), (r2.cand_ids, t2))
        normal = jp.Renderer(new.tok, 100000)
        for q in (CHOICE, NOUL):
            self.assertEqual(normal.render('state', q), new.scorer.render('state', q))

    def test_reverse_key_mapping_and_normalization_in_both_paths(self):
        for prefix in (False, True):
            with self.subTest(prefix=prefix):
                e = engine(reverse=True, prefix=prefix)
                calls = []
                def shared(rendered, ids):
                    calls.append(list(rendered))
                    self.assertEqual(set(rendered), set(ids))
                    return {k: e.scorer.score_rendered([(v[0], v[1])])[0] for k, v in rendered.items()}
                e._score_shared_prefix = shared
                out, raw = e('state', {'c': CHOICE, 'n': NOUL})
                reordered = {**CHOICE, 'criteria': dict(reversed(list(CHOICE['criteria'].items())))}
                out2, _ = e('state', {'c': reordered, 'n': NOUL})
                p = out['answers']['c']['probabilities']
                self.assertEqual(p, out2['answers']['c']['probabilities'])
                self.assertAlmostEqual(sum(p.values()), 1)
                forward, reverse = raw['c']['order_probabilities'].values()
                expected = {k: math.sqrt(forward[k] * reverse[k]) for k in p}
                for k in p:
                    self.assertAlmostEqual(p[k], expected[k] / sum(expected.values()))
                self.assertTrue(raw['c']['reverse_avg'])
                self.assertNotIn('reverse_avg', raw['n'])
                self.assertEqual(sum(t == 'noul' for _, t in e.scorer.seen), 2)
                self.assertEqual(bool(calls), prefix)

    def test_prefix_fallback_and_oom_retry_restore_batch_settings(self):
        for prefix in (False, True):
            e = engine(reverse=True, prefix=prefix)
            e._score_shared_prefix = lambda rendered, ids: None
            score = e.scorer.score_rendered
            attempts = []
            def fail_once(items):
                attempts.append(len(items))
                if len(attempts) == 1:
                    raise OOM()
                self.assertEqual(len(items), 1)
                return score(items)
            e.scorer.score_rendered = fail_once
            out, raw = e('state', {'c': CHOICE, 'n': NOUL})
            self.assertTrue(all(v['oom_retry'] for v in raw.values()))
            self.assertEqual((e.max_batch, e.batch_tokens, e.oom_retries), (8, None, 1))
            self.assertAlmostEqual(sum(out['answers']['c']['probabilities'].values()), 1)
            self.assertEqual(len(attempts), 4)

    def test_shared_prefix_oom_retries_both_orders(self):
        e = engine(reverse=True, prefix=True)
        calls = []
        def shared(rendered, ids):
            calls.append((e.max_batch, e.batch_tokens))
            if len(calls) == 1:
                raise OOM()
            return {k: e.scorer.score_rendered([(v[0], v[1])])[0] for k, v in rendered.items()}
        e._score_shared_prefix = shared
        _, raw = e('state', {'c': CHOICE})
        self.assertEqual(calls, [(8, None), (1, 1)])
        self.assertEqual(len(e.scorer.seen), 2)
        self.assertTrue(raw['c']['oom_retry'])
        self.assertTrue(raw['c']['reverse_avg'])
        self.assertEqual((e.max_batch, e.batch_tokens), (8, None))

    def test_zero_probability_support_is_preserved(self):
        e = engine(reverse=True)
        e.scorer.score_rendered = lambda items: [[0 if k == 'last' else .5 for k in r.keys] for r, _ in items]
        out, _ = e('state', {'c': CHOICE})
        self.assertEqual(out['answers']['c']['probabilities'], {'first': .5, 'best': .5, 'last': 0.0})

    def test_repeat_payload_and_assistant_prefix(self):
        e = engine(repeat=True, reverse=True)
        for q in (CHOICE, NOUL):
            original = engine().scorer.render('UNIQUE_STATE', q).prompt
            rendered = e.scorer.render('UNIQUE_STATE', q)
            payload = original.split('<user>')[1].split('</user>')[0]
            self.assertEqual(rendered.prompt.count(payload), 2)
            self.assertIn(payload + '\n\n' + payload, rendered.prompt)
            self.assertEqual(rendered.prompt.count('<assistant>'), 1)
            self.assertEqual(rendered.cand_ids, engine().scorer.render('UNIQUE_STATE', q).cand_ids)
        _, raw = e('UNIQUE_STATE', {'c': CHOICE, 'n': NOUL})
        self.assertTrue(all(v['repeat_payload'] for v in raw.values()))
        self.assertNotIn('reverse_avg', raw['n'])

    @unittest.skipUnless(os.environ.get('TRACK1_TOKENIZER'), 'set TRACK1_TOKENIZER to a local published tokenizer')
    def test_published_tokenizer_default_and_repeat(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ['TRACK1_TOKENIZER'], local_files_only=True)
        old = jp.Renderer(tok, 100000)
        unchanged = repeat_renderer(jp, tok, 100000, repeat_payload=False)
        repeated = repeat_renderer(jp, tok, 100000, repeat_payload=True)
        for q in (CHOICE, NOUL, {**CHOICE, 'criteria': {'x': None, 'y': 'label'}}):
            a, b, c = [r.render({'facts': 'unique state'}, q) for r in (old, unchanged, repeated)]
            self.assertEqual(a.prompt.encode(), b.prompt.encode())
            self.assertEqual(tok.encode(a.prompt, add_special_tokens=False), tok.encode(b.prompt, add_special_tokens=False))
            self.assertEqual(a.cand_ids, b.cand_ids)
            t = jp.translate_one('q', q)
            payload = jp.build_messages(jp.serialize_state({'facts': 'unique state'}), None, t.question, t.options)[1]['content']
            self.assertEqual(c.prompt.count(payload), 2)
            self.assertTrue(c.prompt.endswith(a.prompt[a.prompt.rfind('<|im_start|>assistant'):]))
            self.assertEqual(c.cand_ids, a.cand_ids)

    def test_actual_shared_prefix_scoring_matches_full_reads(self):
        import torch
        for repeat in (False, True):
            e = engine(reverse=True, repeat=repeat, prefix=True)
            e.torch = torch
            class Body:
                layers = [None]
                def __call__(self, input_ids, past_key_values=None, **kwargs):
                    offset = 0
                    if past_key_values is not None and past_key_values.batch is not None:
                        offset = past_key_values.layers[0].keys
                    summed = input_ids.double().cumsum(dim=1) + offset
                    if past_key_values is not None and past_key_values.batch is None:
                        past_key_values.layers[0].keys = summed[:, -1:]
                    hidden = torch.stack([summed * 1e-5, torch.ones_like(summed)], dim=-1).float()
                    return SimpleNamespace(last_hidden_state=hidden)
            core = SimpleNamespace(model=Body(), lm_head=torch.nn.Linear(2, 256, bias=False))
            with torch.no_grad():
                core.lm_head.weight[:, 0] = torch.arange(256) * .01
                core.lm_head.weight[:, 1] = torch.arange(256) * .001
            e.scorer.model = core
            e.scorer.temperatures = {'choice': 1.2, 'noul': .9}
            e.jm = SimpleNamespace(core_of=lambda x: x, FP32_CANDIDATE_LOGITS=True)
            def score(items):
                out = []
                for r, t in items:
                    ids = torch.tensor([e.tok.encode(r.prompt)])
                    hidden = core.model(input_ids=ids).last_hidden_state[:, -1]
                    candidates = torch.tensor([r.cand_ids])
                    logits = e._candidate_logits(core, hidden, candidates)
                    out.append(torch.softmax(logits[0] / e.scorer.temperatures[t], -1).tolist())
                return out
            e.scorer.score_rendered = score
            questions = {'c': CHOICE, 'n': NOUL}
            shared, _ = e('a sufficiently long shared state ' * 10, questions)
            e.prefix_cache = False
            full, _ = e('a sufficiently long shared state ' * 10, questions)
            for key in CHOICE['criteria']:
                self.assertAlmostEqual(shared['answers']['c']['probabilities'][key], full['answers']['c']['probabilities'][key], places=6)
            self.assertAlmostEqual(shared['answers']['n']['noul'], full['answers']['n']['noul'], places=6)

    def test_repeat_extended_labels(self):
        r = repeat_renderer(jp, Tokenizer(), 100000, repeat_payload=True)
        p = r.render_messages('s', 'q', ['one', 'two'], ['X', 'Y'])
        self.assertEqual(p.count('X. one'), 2)
        self.assertEqual(p.count('Y. two'), 2)
        self.assertEqual(p.count('<assistant>'), 1)


if __name__ == '__main__':
    unittest.main()
