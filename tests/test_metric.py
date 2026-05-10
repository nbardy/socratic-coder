from __future__ import annotations

import random

import numpy as np

from user_mimic.metric import Score, score


class FixedJudge:
    def __init__(self, response: str):
        self._response = response
        self.calls: list[str] = []

    def compare(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class RaisingJudge:
    def compare(self, prompt: str) -> str:
        raise RuntimeError("judge down")


class SlotAwareJudge:
    """Always picks whichever slot contains the target substring."""

    def __init__(self, target_substring: str):
        self._target = target_substring

    def compare(self, prompt: str) -> str:
        a_line = [ln for ln in prompt.splitlines() if ln.startswith("A) ")][0]
        return "A" if self._target in a_line else "B"


class FixedEmbedder:
    def __init__(self, vectors: list[list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == len(self._vectors)
        return np.asarray(self._vectors, dtype=np.float32)


class RaisingEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("embed down")


CTX = [{"role": "user", "content": "What should we do next?"}]


def test_score_judge_picks_generated_gives_1():
    judge = SlotAwareJudge(target_substring="GENERATED_REPLY")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("GENERATED_REPLY", "gold reply text", CTX, judge, embedder,
              rng=random.Random(0))
    assert s.judge == 1.0


def test_score_judge_picks_gold_gives_0():
    judge = SlotAwareJudge(target_substring="GOLD_UNIQUE")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("generated", "GOLD_UNIQUE", CTX, judge, embedder, rng=random.Random(0))
    assert s.judge == 0.0


def test_score_embed_uses_clipped_cosine():
    judge = FixedJudge("A")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("gen", "gold", CTX, judge, embedder, rng=random.Random(0))
    assert s.embed == 1.0

    embedder2 = FixedEmbedder([[1.0, 0.0], [-1.0, 0.0]])
    s2 = score("gen", "gold", CTX, judge, embedder2, rng=random.Random(0))
    assert s2.embed == 0.0


def test_length_score_bands():
    judge = FixedJudge("A")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    gold = "abcdefghij"

    s_eq = score(gold, gold, CTX, judge, embedder, rng=random.Random(0))
    assert s_eq.length == 1.0

    s_2x = score(gold * 2, gold, CTX, judge, embedder, rng=random.Random(0))
    assert s_2x.length == 1.0

    s_4x = score(gold * 4, gold, CTX, judge, embedder, rng=random.Random(0))
    assert 0.0 <= s_4x.length < 1.0

    s_huge = score(gold * 1024, gold, CTX, judge, embedder, rng=random.Random(0))
    assert s_huge.length == 0.0


def test_judge_failure_scores_half():
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("gen", "gold", CTX, RaisingJudge(), embedder, rng=random.Random(0))
    assert s.judge == 0.5


def test_embed_failure_scores_zero():
    judge = FixedJudge("A")
    s = score("gen", "gold", CTX, judge, RaisingEmbedder(), rng=random.Random(0))
    assert s.embed == 0.0


def test_combined_is_weighted_sum():
    judge = SlotAwareJudge(target_substring="GEN_UNIQ")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("GEN_UNIQ", "gold text", CTX, judge, embedder, rng=random.Random(0))
    expected = 0.6 * s.judge + 0.3 * s.embed + 0.1 * s.length
    assert abs(s.combined - expected) < 1e-9
    assert isinstance(s, Score)


def test_judge_parser_accepts_full_sentence():
    judge = FixedJudge("B) is more in the author's style because...")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("gen", "gold", CTX, judge, embedder, rng=random.Random(0))
    assert s.judge in (0.0, 1.0)


def test_judge_parser_rejects_garbage_as_tie():
    judge = FixedJudge("...............")
    embedder = FixedEmbedder([[1.0, 0.0], [1.0, 0.0]])
    s = score("gen", "gold", CTX, judge, embedder, rng=random.Random(0))
    assert s.judge == 0.5
