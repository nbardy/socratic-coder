from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nick_mimic.canonical import Sample
from nick_mimic.template import JudgePrompt, render_for_judge


class JudgeLM(Protocol):
    def compare(self, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Score:
    judge: float
    embed: float
    length: float
    combined: float


_FIRST_AB = re.compile(r"[AaBb]")


def _parse_ab(response: str) -> str | None:
    s = response.strip()
    if not s:
        return None
    m = _FIRST_AB.search(s)
    if m is None:
        return None
    return m.group(0).upper()


def _judge_score(judge_choice: str | None, gold_is: str) -> float:
    if judge_choice is None:
        return 0.5
    if judge_choice == gold_is:
        return 0.0
    return 1.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embed_score(generated: str, gold: str, embedder: Embedder) -> float:
    vecs = embedder.embed([generated, gold])
    sim = _cosine(vecs[0], vecs[1])
    return float(max(0.0, min(1.0, sim)))


def _length_score(generated: str, gold: str) -> float:
    ratio = len(generated) / max(len(gold), 1)
    if ratio <= 0.0:
        return 0.0
    dist = abs(math.log(ratio) / math.log(2.0))
    return float(max(0.0, min(1.0, 1.0 - max(0.0, dist - 1.0))))


def _safe_judge(judge: JudgeLM, jp: JudgePrompt) -> float:
    try:
        raw = judge.compare(jp.prompt)
    except Exception:
        return 0.5
    return _judge_score(_parse_ab(raw), jp.gold_is)


def score(
    generated: str,
    gold: str,
    context: list[dict],
    judge: JudgeLM,
    embedder: Embedder,
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    *,
    sample: Sample | None = None,
    rng: random.Random | None = None,
) -> Score:
    judge_sample = sample if sample is not None else _ephemeral_sample(context)
    jp = render_for_judge(judge_sample, generated, gold, rng=rng)
    j = _safe_judge(judge, jp)
    try:
        e = _embed_score(generated, gold, embedder)
    except Exception:
        e = 0.0
    ln = _length_score(generated, gold)
    wj, we, wl = weights
    combined = wj * j + we * e + wl * ln
    return Score(judge=j, embed=e, length=ln, combined=combined)


def _ephemeral_sample(context: list[dict]) -> Sample:
    return Sample(
        context_messages=list(context),
        target="",
        project="",
        harness="",
        model="",
        cwd="",
        source_path="",
        thread_id="",
    )


class OpenAIJudge:
    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> "OpenAIJudge":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot construct OpenAIJudge.from_env()")
        from openai import OpenAI

        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        return cls(OpenAI(api_key=api_key), model)

    def compare(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


class OpenAIEmbedder:
    def __init__(self, client, model: str = "text-embedding-3-small"):
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> "OpenAIEmbedder":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot construct OpenAIEmbedder.from_env()")
        from openai import OpenAI

        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return cls(OpenAI(api_key=api_key), model)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
