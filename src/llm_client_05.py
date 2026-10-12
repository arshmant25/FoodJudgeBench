"""Provider-agnostic chat client for OpenAI-compatible endpoints (free tiers), plus an offline mock.

Features: on-disk response cache (re-runs and resumes cost nothing), requests-per-minute AND tokens-per-minute throttling,
retry with back-off on 429/5xx, a clean `QuotaExhausted` stop when a daily quota is hit, optional provider-specific
request fields, and `<think>` stripping for reasoning models. API keys come from environment variables or a `.env` file
in the project root (never written to the cache).

Smoke test a provider/model before a long run:
    python src/llm_client.py --provider groq --model openai/gpt-oss-120b --ping
Free-tier limits and model names change often: verify them on the provider's page; defaults here are only a starting point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "processed" / "llm_cache"

# base URLs: Hugging Face's is from its docs; the others are the providers' documented OpenAI-compatible endpoints
# (not testable from the build sandbox - use --ping). default_model=None means "you must pass --model".
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "default_model": "openai/gpt-oss-120b",
             "rpm": 25, "tpm": 7000},          # free plan (docs): 30 RPM, 8K TPM, 1K RPD, 200K TPD for gpt-oss-120b
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY",
               "default_model": "gemini-2.5-flash", "rpm": 8, "tpm": None},   # free limits are shown in AI Studio, not published
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY", "default_model": None,
                   "rpm": 18, "tpm": None},    # ':free' models: 20 RPM and 50 requests/day (1000/day after a $10 top-up)
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY", "default_model": None,
               "rpm": 35, "tpm": None},        # NIM free tier reported ~40 RPM; needs NVIDIA developer membership
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "key_env": "CEREBRAS_API_KEY", "default_model": None, "rpm": 25, "tpm": None},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY", "default_model": None, "rpm": 20, "tpm": None},
    "huggingface": {"base_url": "https://router.huggingface.co/v1", "key_env": "HF_TOKEN", "default_model": "openai/gpt-oss-120b",
                    "rpm": 20, "tpm": None},   # NOT free: needs purchased credits or PRO (see docs); model suffix picks provider
    # local Ollama (OpenAI-compatible endpoint, no key, no quota): set OLLAMA_HOST to use another machine, e.g. http://192.168.1.5:11434
    "ollama": {"base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/").removesuffix("/v1") + "/v1", "key_env": None,
               "default_model": None, "rpm": 600, "tpm": None},
    "custom": {"base_url": os.environ.get("FJB_BASE_URL", ""), "key_env": "FJB_API_KEY", "default_model": None, "rpm": 20, "tpm": None},
}


class QuotaExhausted(RuntimeError):
    """Daily/monthly quota hit (or a wait longer than max_wait was requested). Progress is saved; re-run later."""


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    cached: bool = False
    latency: float = 0.0


def load_env_file(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE per line); real environment variables win."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return re.sub(r"^.*?</think>", "", text, flags=re.S | re.I).strip() if "</think>" in text else text.strip()


class _Throttle:
    """Sliding-window limiter for requests/minute and tokens/minute."""

    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self.events: deque = deque()          # (timestamp, tokens)

    def wait(self, est_tokens: int) -> None:
        while True:
            now = time.time()
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()
            n_req = len(self.events)
            n_tok = sum(t for _, t in self.events)
            need = 0.0
            if self.rpm and n_req >= self.rpm:
                need = max(need, 60 - (now - self.events[0][0]))
            if self.tpm and self.events and n_tok + est_tokens > self.tpm:
                need = max(need, 60 - (now - self.events[0][0]))
            if need <= 0:
                return
            time.sleep(min(need + 0.05, 30))

    def record(self, tokens: int) -> None:
        self.events.append((time.time(), tokens))


class LLMClient:
    def __init__(self, provider: str, model: str | None = None, rpm: int | None = None, tpm: int | None = None,
                 extra: dict | None = None, max_wait: float = 120.0, cache: bool = True, timeout: float = 180.0):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider '{provider}'. choose from {sorted(PROVIDERS) + ['mock']}")
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.model = model or os.environ.get("FJB_MODEL") or cfg["default_model"]
        if not self.model:
            raise ValueError(f"provider '{provider}' has no default model: pass --model (see the provider's model list)")
        self.base_url = cfg["base_url"].rstrip("/")
        if not self.base_url:
            raise ValueError("custom provider needs FJB_BASE_URL")
        self.key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else "ollama"     # Ollama ignores the key
        if provider == "ollama":
            timeout = max(timeout, 900.0)                          # local models can take minutes per call on CPU
        if not self.key:
            raise RuntimeError(f"missing API key: set {cfg['key_env']} in the environment or in {ROOT / '.env'}")
        self.extra = extra or {}
        self.max_wait, self.use_cache, self.timeout = max_wait, cache, timeout
        self.throttle = _Throttle(rpm if rpm is not None else cfg["rpm"], tpm if tpm is not None else cfg["tpm"])
        self.calls = self.cache_hits = 0
        self.tokens_in = self.tokens_out = 0

    # ------------------------------------------------------------------
    def _key(self, body: dict) -> str:
        blob = json.dumps({"p": self.provider, "b": {k: v for k, v in body.items()}}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, task: str = "generic", temperature: float = 0.4, max_tokens: int = 900, meta=None) -> LLMResult:
        """`task` and `meta` are only used by the offline mock; real providers ignore them."""
        import requests

        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **self.extra}
        ck = self._key(body)
        cpath = CACHE_DIR / self.provider / f"{ck}.json"
        if self.use_cache and cpath.exists():
            d = json.loads(cpath.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return LLMResult(d["text"], d["model"], d.get("usage", {}), True, 0.0)

        est = sum(len(m["content"]) for m in messages) // 3 + min(max_tokens, 500)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        tweaked: set[str] = set()
        for attempt in range(7):
            self.throttle.wait(est)
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                self._sleep_or_raise(min(60, 2 ** attempt + random.random()), f"network error: {e}", attempt, 5)
                continue
            self.throttle.record(est)
            if r.status_code == 200:
                data = r.json()
                try:
                    text = strip_thinking(data["choices"][0]["message"]["content"] or "")
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError(f"unexpected response shape: {str(data)[:300]}")
                usage = data.get("usage") or {}
                self.calls += 1
                self.tokens_in += usage.get("prompt_tokens", 0)
                self.tokens_out += usage.get("completion_tokens", 0)
                if usage.get("total_tokens"):                      # replace the estimate with the real figure
                    self.throttle.events[-1] = (self.throttle.events[-1][0], usage["total_tokens"])
                if not text:
                    self._sleep_or_raise(2 + attempt, "empty completion (reasoning budget used up? raise --max-tokens)", attempt, 3)
                    continue
                res = LLMResult(text, data.get("model", self.model), usage, False, time.time() - t0)
                if self.use_cache:
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                    cpath.write_text(json.dumps({"text": text, "model": res.model, "usage": usage}, ensure_ascii=False), encoding="utf-8")
                return res
            if r.status_code == 429:
                wait = _retry_after(r) or min(60, 2 ** attempt * 2 + random.random())
                low = r.text.lower()
                if wait > self.max_wait or any(w in low for w in ("per day", "daily", "tpd", "rpd", "quota exceeded")) and wait > 30:
                    raise QuotaExhausted(f"{self.provider}: rate limit says wait {wait:.0f}s ({r.text[:200]})")
                self._sleep_or_raise(wait, f"429 rate limited ({r.text[:120]})", attempt, 6)
                continue
            if r.status_code in (401, 403):
                raise RuntimeError(f"{self.provider}: authentication failed ({r.status_code}): {r.text[:200]}")
            if r.status_code == 400:                                # unsupported parameter: adapt once each, else give up
                low = r.text.lower()
                if "max_tokens" in low and "max_completion_tokens" in low and "max_tokens" in body and "mct" not in tweaked:
                    body["max_completion_tokens"] = body.pop("max_tokens"); tweaked.add("mct"); continue
                if "temperature" in low and "temperature" in body and "temp" not in tweaked:
                    body.pop("temperature"); tweaked.add("temp"); continue
                raise RuntimeError(f"{self.provider}: bad request: {r.text[:300]}")
            if r.status_code == 404:
                raise RuntimeError(f"{self.provider}: not found (404) - check the model name '{self.model}': {r.text[:200]}")
            self._sleep_or_raise(min(60, 2 ** attempt + random.random()), f"HTTP {r.status_code}: {r.text[:120]}", attempt, 5)
        raise RuntimeError(f"{self.provider}: gave up after repeated failures")

    def _sleep_or_raise(self, wait: float, why: str, attempt: int, max_attempts: int) -> None:
        if attempt + 1 >= max_attempts:
            raise RuntimeError(f"{self.provider}: {why} (after {attempt + 1} attempts)")
        print(f"    [{self.provider}] {why}; waiting {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)

    def stats(self) -> dict:
        return {"provider": self.provider, "model": self.model, "api_calls": self.calls, "cache_hits": self.cache_hits,
                "prompt_tokens": self.tokens_in, "completion_tokens": self.tokens_out}


def _retry_after(r) -> float | None:
    v = r.headers.get("retry-after") or r.headers.get("Retry-After")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


# ============================================================================== offline mock (for dry runs and tests)
class MockClient:
    """Deterministic stand-in: exercises the whole pipeline (including the check/revise loop) without any API.
    It deliberately writes one invented sentence into about half of the grounded drafts and omits a must-mention fact
    in about a third, so the self-check technique has something to find."""

    provider, model = "mock", "mock-1"

    def __init__(self, *a, **k):
        self.calls = self.cache_hits = 0
        self.tokens_in = self.tokens_out = 0

    def stats(self) -> dict:
        return {"provider": "mock", "model": "mock-1", "api_calls": self.calls, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, task="generic", temperature=0.4, max_tokens=900, meta=None) -> LLMResult:
        self.calls += 1
        meta = meta or {}
        h = int(hashlib.md5(json.dumps(meta.get("title", "") + str(meta.get("case", "")), default=str).encode()).hexdigest(), 16)
        facts = meta.get("facts", [])
        if task == "explain_direct":
            ings = ", ".join(meta.get("ingredients", [])[:3])
            text = (f"{meta['title']} looks like a good match for you. It uses {ings} and takes about {meta.get('minutes', 30)} minutes, "
                    f"so it should fit a busy evening. Many people enjoy it as a relaxed family dinner, and it is packed with flavour.")
        elif task == "explain_grounded":
            by = {f["id"]: f for f in facts}
            recipe = next(f for f in facts if f["kind"] == "recipe")
            parts = [f"The recipe is {meta['title']} [{recipe['id']}]."]
            musts = [f for f in facts if f["must_mention"]]
            if h % 3 == 0 and len(musts) > 0:
                musts = musts[1:]                                           # omit one must-mention fact
            for f in facts:
                if f["kind"] == "match" and not f["must_mention"]:
                    parts.append(f"{f['text'].rstrip('.')} [{f['id']}].")
            for f in musts:
                parts.append(f"A trade-off to know: {f['text'].rstrip('.')} [{f['id']}].")
            if h % 2 == 0:
                parts.append(f"It is also great for family dinners [{recipe['id']}].")
            text = " ".join(parts[:7])
        elif task == "check":
            sents = re.split(r"(?<=[.!?])\s+", meta.get("text", ""))
            claims = []
            for s in sents:
                bad = "family" in s.lower()
                claims.append({"claim": s, "verdict": "unsupported" if bad else "supported", "fact_ids": [] if bad else ["F1"]})
            text = "```json\n" + json.dumps({"claims": claims}) + "\n```"
        elif task == "revise":
            sents = re.split(r"(?<=[.!?])\s+", meta.get("text", ""))
            keep = [s for s in sents if "family" not in s.lower()]
            by = {f["id"]: f for f in facts}
            for fid in meta.get("missing_ids", []):
                keep.append(f"A trade-off to know: {by[fid]['text'].rstrip('.')} [{fid}].")
            text = " ".join(keep)
        elif task == "shorten":
            text = " ".join(meta.get("text", "").split()[:90])
        else:
            text = "ok"
        self.tokens_in += sum(len(m["content"]) for m in messages) // 4
        self.tokens_out += len(text) // 4
        return LLMResult(text, "mock-1", {}, False, 0.0)


def get_client(provider: str, model: str | None = None, **kw):
    load_env_file()
    if provider == "mock":
        return MockClient()
    return LLMClient(provider, model, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test an LLM provider")
    ap.add_argument("--provider", default=os.environ.get("FJB_PROVIDER", "groq"))
    ap.add_argument("--model")
    ap.add_argument("--extra", help='JSON of extra request fields, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--ping", action="store_true")
    a = ap.parse_args()
    c = get_client(a.provider, a.model, extra=json.loads(a.extra) if a.extra else None, cache=False)
    t0 = time.time()
    r = c.chat([{"role": "user", "content": "Reply with exactly one word: ok"}], max_tokens=300, temperature=0)
    print(f"{c.provider} / {r.model}: reply={r.text!r} in {time.time() - t0:.1f}s usage={r.usage}")


if __name__ == "__main__":
    main()