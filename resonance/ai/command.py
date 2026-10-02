# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/command.py -- natural-language control of the engine.

HYPOTHESIS
    "Make it darker" is a real request with a real DSP answer --
    it should not need a cloud round-trip. So the local parser
    handles the dozen most common moves deterministically
    (offline, instant, testable), and the Gemini parser exists
    for freeform phrasing the local one cannot cover. Both
    produce the same Intent; the executor only runs known
    actions with sane parameters, and refuses everything else
    loudly instead of guessing.

METHOD
    parse_command(text): regex/keyword Intents with confidence
    and a plain-English explanation of the mapping. Unmatched
    text -> action "unknown", confidence 0.0 (never a guess).
    ai_parse(text, api_key): Gemini constrained to emit JSON in
    the known action schema; the result is validated fail-closed
    (unknown action or bad params -> ValueError, not execution).
    execute_intent(intent, audio, sr): runs the real engine --
    core.edit biquad/gain/normalize/limiter/fades, quantize,
    autotune -- and returns (audio_out, report_text).

RESULT
    "make it darker" -> lowpass @ 4 kHz, and the report says so.
    "tighten the timing at 120 bpm" -> quantize strength 0.7 @
    120 BPM, with the QuantizeReport attached. What the engine
    did is always printed; silent enhancement stays forbidden.
"""
import json
import re
from dataclasses import dataclass, field

import numpy as np

from resonance.core import edit as _edit
from resonance.quantize import quantize as _quantize
from resonance.pitch import autotune as _autotune

# Every action the executor knows. The Gemini parser is constrained
# to this set; anything else is rejected, not attempted.
KNOWN_ACTIONS = (
    "darken",        # lowpass: reduce treble
    "brighten",      # peaking boost at 8 kHz
    "warm",          # gentle low shelf-ish: peaking +3 dB at 200 Hz
    "tighten_timing",  # quantize to the beat grid (needs bpm)
    "louder",        # gain up
    "quieter",       # gain down
    "normalize",     # normalize to 0.89
    "limit",         # soft limiter
    "fade_in",       # fade in over N seconds
    "fade_out",      # fade out over N seconds
    "tune",          # corrective autotune, chromatic
)


@dataclass
class Intent:
    """One parsed command: what to do, with what, how sure, from where."""
    action: str
    params: dict = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "local"
    explanation: str = ""

    @property
    def known(self):
        return self.action in KNOWN_ACTIONS


def _db_amount(text, default=6.0):
    if re.search(r"\b(a bit|slightly|a little|gently)\b", text):
        return 3.0
    if re.search(r"\b(a lot|much|heavily)\b", text):
        return 12.0
    return default


def _seconds(text, default=2.0):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|second)", text)
    if m:
        return max(0.1, float(m.group(1)))
    return default


def _bpm(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*bpm", text)
    if m:
        bpm = float(m.group(1))
        if 20.0 <= bpm <= 300.0:
            return bpm
    return None


def parse_command(text):
    """Deterministic NL -> Intent. Offline. Never guesses.

    Covers the common moves; anything else returns action
    "unknown" with confidence 0.0 and an explanation saying so.
    """
    t = text.strip().lower()
    if not t:
        return Intent(action="unknown", explanation="empty command")

    if re.search(r"\b(darker|dark|mellow|dull the|take the edge off)\b", t):
        return Intent("darken", {"freq": 4000.0}, 0.95, "local",
                      "darken = lowpass at 4 kHz (treble reduced)")
    if re.search(r"\b(brighter|bright|airy|presence|sparkle)\b", t):
        return Intent("brighten", {"freq": 8000.0, "gain_db": 4.0}, 0.9,
                      "local", "brighten = +4 dB peaking at 8 kHz")
    if re.search(r"\b(warm|warmer|warmth|body|thicken)\b", t):
        return Intent("warm", {"freq": 200.0, "gain_db": 3.0}, 0.9, "local",
                      "warm = +3 dB peaking at 200 Hz")
    if re.search(r"\b(tighten|quantize|snap|on the grid|to the grid)\b", t):
        bpm = _bpm(t)
        strength = 0.3 if re.search(
            r"\b(a bit|slightly|gently)\b", t) else 0.7
        params = {"strength": strength}
        conf, expl = 0.9, f"tighten_timing = quantize, strength {strength}"
        if bpm is not None:
            params["bpm"] = bpm
            expl += f" at {bpm:g} BPM"
        else:
            conf = 0.55
            expl += " (no BPM given -- supply one with e.g. 'at 120 bpm')"
            params["bpm"] = None
        return Intent("tighten_timing", params, conf, "local", expl)
    if re.search(r"\b(normali[sz]e)\b", t):
        return Intent("normalize", {"target": 0.89}, 0.95, "local",
                      "normalize = peak normalize to 0.89")
    if re.search(r"\b(limit|limiter|master|glue)\b", t):
        return Intent("limit", {"ceiling": 0.98}, 0.9, "local",
                      "limit = soft limiter, ceiling 0.98")
    if re.search(r"\b(louder|turn (it |this )?up|amplify|boost (the )?volume)\b", t):
        db = _db_amount(t)
        return Intent("louder", {"db": db}, 0.9, "local",
                      f"louder = +{db:g} dB gain")
    if re.search(r"\b(quieter|turn (it |this )?down|reduce (the )?volume)\b", t):
        db = _db_amount(t)
        return Intent("quieter", {"db": db}, 0.9, "local",
                      f"quieter = -{db:g} dB gain")
    if re.search(r"\bfade.?in\b", t):
        s = _seconds(t)
        return Intent("fade_in", {"seconds": s}, 0.9, "local",
                      f"fade_in over {s:g} s")
    if re.search(r"\bfade.?out\b", t):
        s = _seconds(t)
        return Intent("fade_out", {"seconds": s}, 0.9, "local",
                      f"fade_out over {s:g} s")
    if re.search(r"\b(tune|autotune|auto-tune|pitch.?correct)\b", t):
        return Intent("tune", {"scale": "chromatic", "mode": "corrective"},
                      0.9, "local",
                      "tune = corrective autotune to chromatic")
    return Intent(
        action="unknown", confidence=0.0, source="local",
        explanation=f"no local mapping for {text.strip()!r} -- try "
                    f"'make it darker', 'tighten the timing at 120 bpm', "
                    f"'louder', or use the --ai parser for freeform phrasing")


_AI_PARSE_SYSTEM = (
    "You translate a producer's plain-English request into ONE JSON "
    "object with keys: action (one of: " + ", ".join(KNOWN_ACTIONS) + "), "
    "params (object), confidence (0-1), explanation (one sentence). "
    "Valid params per action: darken {freq}, brighten {freq, gain_db}, "
    "warm {freq, gain_db}, tighten_timing {bpm (required, number), "
    "strength (0-1)}, louder {db}, quieter {db}, normalize {target}, "
    "limit {ceiling}, fade_in {seconds}, fade_out {seconds}, "
    "tune {scale, mode}. If the request maps to nothing, use "
    "action \"unknown\" with confidence 0. Reply with ONLY the JSON.")


def ai_parse(text, api_key=None, model=None):
    """Gemini-backed freeform parse. Validated fail-closed.

    Raises GeminiKeyMissing without a key; ValueError when the
    model returns an action outside KNOWN_ACTIONS or malformed
    params -- the executor never sees an unvalidated intent.
    """
    from resonance.ai.gemini import DEFAULT_MODEL, GeminiClient
    client = GeminiClient(api_key=api_key, model=model or DEFAULT_MODEL)
    result = client.generate(
        _AI_PARSE_SYSTEM + "\n\nRequest: " + text.strip(),
        think_budget=2048, code_execution=False, max_tokens=512)
    try:
        data = json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ai_parse: model did not return JSON: {exc}")
    action = data.get("action")
    if action not in KNOWN_ACTIONS and action != "unknown":
        raise ValueError(f"ai_parse: unknown action {action!r} rejected")
    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("ai_parse: params must be an object")
    if action == "tighten_timing":
        bpm = params.get("bpm")
        if not isinstance(bpm, (int, float)) or not 20.0 <= bpm <= 300.0:
            raise ValueError("ai_parse: tighten_timing needs a BPM 20-300")
        strength = params.get("strength", 0.7)
        if not isinstance(strength, (int, float)) or not 0.0 <= strength <= 1.0:
            raise ValueError("ai_parse: strength must be 0-1")
    return Intent(action=action, params=params,
                  confidence=float(data.get("confidence", 0.5)),
                  source="gemini",
                  explanation=str(data.get("explanation", "")))


def execute_intent(intent, audio, sr):
    """Run an Intent through the real engine.

    Returns (audio_out, report_text). Raises ValueError for
    unknown actions -- nothing executes on a guess.
    """
    if not intent.known:
        raise ValueError(
            f"execute_intent: refusing unknown action {intent.action!r} "
            f"({intent.explanation})")
    sr = int(sr)
    p = intent.params
    action = intent.action
    if action == "darken":
        out = _edit.biquad(audio, "lowpass", float(p.get("freq", 4000.0)),
                           sample_rate=sr)
        report = f"darken: lowpass at {p.get('freq', 4000.0):g} Hz"
    elif action == "brighten":
        out = _edit.biquad(audio, "peaking", float(p.get("freq", 8000.0)),
                           gain_db=float(p.get("gain_db", 4.0)),
                           sample_rate=sr)
        report = (f"brighten: +{p.get('gain_db', 4.0):g} dB peaking "
                  f"at {p.get('freq', 8000.0):g} Hz")
    elif action == "warm":
        out = _edit.biquad(audio, "peaking", float(p.get("freq", 200.0)),
                           gain_db=float(p.get("gain_db", 3.0)),
                           sample_rate=sr)
        report = (f"warm: +{p.get('gain_db', 3.0):g} dB peaking "
                  f"at {p.get('freq', 200.0):g} Hz")
    elif action == "tighten_timing":
        bpm = p.get("bpm")
        if bpm is None:
            raise ValueError("execute_intent: tighten_timing needs a BPM "
                             "(e.g. 'tighten the timing at 120 bpm')")
        out, qreport = _quantize(audio, sr, float(bpm),
                                 strength=float(p.get("strength", 0.7)))
        report = (f"tighten_timing: quantized at {float(bpm):g} BPM, "
                  f"strength {float(p.get('strength', 0.7))}\n"
                  f"{qreport.summary() if hasattr(qreport, 'summary') else qreport}")
    elif action == "louder":
        out = _edit.gain(audio, float(p.get("db", 6.0)), unit="db")
        report = f"louder: +{float(p.get('db', 6.0)):g} dB"
    elif action == "quieter":
        out = _edit.gain(audio, -float(p.get("db", 6.0)), unit="db")
        report = f"quieter: -{float(p.get('db', 6.0)):g} dB"
    elif action == "normalize":
        out = _edit.normalize(audio, target=float(p.get("target", 0.89)))
        report = f"normalize: peak to {float(p.get('target', 0.89)):g}"
    elif action == "limit":
        out = _edit.soft_limiter(audio, ceiling=float(p.get("ceiling", 0.98)))
        report = f"limit: soft limiter, ceiling {float(p.get('ceiling', 0.98)):g}"
    elif action == "fade_in":
        n = int(float(p.get("seconds", 2.0)) * sr)
        out = _edit.fade_in(audio, n, kind="linear")
        report = f"fade_in: {float(p.get('seconds', 2.0)):g} s linear"
    elif action == "fade_out":
        n = int(float(p.get("seconds", 2.0)) * sr)
        out = _edit.fade_out(audio, n, kind="linear")
        report = f"fade_out: {float(p.get('seconds', 2.0)):g} s linear"
    elif action == "tune":
        out, treport = _autotune(audio, sr, scale=p.get("scale", "chromatic"),
                                 mode=p.get("mode", "corrective"))
        report = f"tune: {treport.summary()}"
    else:  # pragma: no cover -- guarded by intent.known
        raise ValueError(f"execute_intent: unhandled action {action!r}")
    return np.asarray(out, dtype=np.float32), report
