# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/gemini.py -- the cloud guest: Gemini API client.

HYPOTHESIS
    Cloud AI is a guest in the wing, not a resident: it must never
    be required, never phone home silently, and never hold a
    credential at rest. So the key arrives at runtime -- explicit
    argument or the GEMINI_API_KEY environment variable -- and is
    held in memory only for the duration of the call. No key, no
    call: GeminiKeyMissing names exactly how to provide one.

METHOD
    Stdlib urllib only (the repo's numpy-only rule holds). POST to
    generativelanguage.googleapis.com/v1beta/models/<model>:
    generateContent with an optional thinking budget ("extended
    complexity": thinkingConfig.thinkingBudget) and the code-
    execution tool. Response parts are walked honestly: thought
    parts are skipped, executableCode/codeExecutionResult are
    reported, text parts are concatenated. HTTP errors surface as
    GeminiError carrying the status and the provider's message --
    never swallowed, never retried silently.

RESULT
    GeminiClient(api_key=None) resolves the key, generate() returns
    a GeminiResult(text=..., code_executed=bool, usage=dict).
    Without a key every cloud path raises before any byte leaves
    the machine.

SECURITY
    The key is never written to disk, never logged, never included
    in exceptions or usage reports. Tests exercise key resolution,
    request-body construction, and response parsing against canned
    payloads -- no network in the suite.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

API_HOST = "generativelanguage.googleapis.com"
API_BASE = f"https://{API_HOST}/v1beta/"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_KEY_ENV = "GEMINI_API_KEY"


class GeminiKeyMissing(Exception):
    """Raised before any network I/O when no API key is available."""


class GeminiError(Exception):
    """A failed Gemini call: carries HTTP status + provider message."""

    def __init__(self, status, message):
        super().__init__(f"Gemini API error {status}: {message}")
        self.status = status
        self.provider_message = message


class GeminiResult:
    """What came back: text, whether code ran, token usage."""

    def __init__(self, text, code_executed=False, usage=None):
        self.text = text
        self.code_executed = code_executed
        self.usage = usage or {}

    def __repr__(self):
        return (f"GeminiResult(chars={len(self.text)}, "
                f"code_executed={self.code_executed})")


def resolve_key(api_key=None, env_var=DEFAULT_KEY_ENV):
    """Find the key: explicit argument wins, else the env var.

    Raises GeminiKeyMissing (naming the env var) when neither exists.
    The returned key is never logged by this module.
    """
    if api_key:
        return api_key
    found = os.environ.get(env_var)
    if found:
        return found
    raise GeminiKeyMissing(
        f"No Gemini API key. Pass api_key=... or set the {env_var} "
        f"environment variable. Get a key at https://aistudio.google.com "
        f"(free tier). The key is used in memory only and never stored."
    )


def build_request_body(prompt, think_budget=0, code_execution=False,
                       max_tokens=8192):
    """Build the generateContent body (pure function: testable offline)."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": int(max_tokens)},
    }
    if think_budget and think_budget > 0:
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": int(think_budget),
            "includeThoughts": True,
        }
    if code_execution:
        body["tools"] = [{"codeExecution": {}}]
    return body


def parse_response(payload):
    """Walk a generateContent response dict into a GeminiResult.

    Thought parts are skipped; code execution is reported, not
    hidden; text parts concatenate in order. A blocked/empty
    response raises GeminiError naming the finish reason.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        raise GeminiError(
            0, "empty response: no candidates "
               f"(promptFeedback={payload.get('promptFeedback')})")
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    texts, code_executed = [], False
    for part in parts:
        if part.get("thought"):
            continue
        if "executableCode" in part or "codeExecutionResult" in part:
            code_executed = True
            continue
        if "text" in part:
            texts.append(part["text"])
    text = "".join(texts).strip()
    if not text and not code_executed:
        raise GeminiError(
            0, f"no text in response (finishReason={cand.get('finishReason')})")
    usage = payload.get("usageMetadata") or {}
    return GeminiResult(text=text, code_executed=code_executed,
                        usage=dict(usage))


class GeminiClient:
    """A Gemini API client that never stores its key.

    api_key: explicit key (wins) or None to read GEMINI_API_KEY.
    model:   API model id; default gemini-3.8-flash (verified live
             with thinking + code execution).
    """

    def __init__(self, api_key=None, model=DEFAULT_MODEL,
                 key_env=DEFAULT_KEY_ENV, timeout=120):
        self._key = resolve_key(api_key, env_var=key_env)
        self.model = model
        self.timeout = timeout

    def _post(self, body):
        url = (API_BASE + "models/" + urllib.parse.quote(self.model, safe="")
               + ":generateContent")
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self._key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.load(exc)
                message = (detail.get("error") or {}).get("message", "")
            except Exception:
                message = exc.reason
            raise GeminiError(exc.code, message or str(exc.reason))

    def generate(self, prompt, think_budget=4096, code_execution=False,
                 max_tokens=8192):
        """One completion. Key was resolved at construction; the only
        network call this client ever makes happens here."""
        body = build_request_body(prompt, think_budget=think_budget,
                                  code_execution=code_execution,
                                  max_tokens=max_tokens)
        return parse_response(self._post(body))
