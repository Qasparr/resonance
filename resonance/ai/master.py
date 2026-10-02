# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/master.py -- AI-assisted mastering.

HYPOTHESIS
    Mastering advice is two jobs with different honesty
    requirements: the LEVEL job (clipping, headroom, DC) is
    arithmetic and belongs offline; the TASTE job ("warmer",
    "more glue", "competitive loudness") is judgment and is
    where the cloud guest earns its keep. So auto_chain() is a
    deterministic rule set over measured numbers -- no model --
    while advise() sends the measurement report plus the user's
    goal to Gemini and returns its words, labeled as advice,
    never applied silently.

METHOD
    auto_chain(analysis): maps flags to core.edit.apply_chain
    effect dicts (limiter when pinned, normalize when faint,
    highpass when DC-offset). Every rule names the measurement
    that triggered it; an empty chain means "nothing wrong found",
    and the report says so.
    advise(analysis, goal, api_key=None): builds the prompt from
    format_report() + goal, calls GeminiClient with extended
    thinking + code execution, returns GeminiResult. No key ->
    GeminiKeyMissing before any network I/O.

RESULT
    The AI suggests; the engine applies. apply_chain() from
    core.edit runs whatever chain the user approves -- the
    module never writes audio by itself.
"""
from resonance.ai.analyze import AnalysisResult, format_report
from resonance.ai.gemini import GeminiClient, GeminiResult

# Rule thresholds: conservative, documented, user-overridable by
# simply editing the chain before applying it.
_CLIP_FRACTION_LIMITER = 0.0      # any clipping -> limiter
_HOT_RMS_DB = -3.0                # RMS within 3 dB of FS -> limiter
_FAINT_RMS_DB = -24.0             # quieter than this -> normalize up
_DC_OFFSET_LIMIT = 0.01
_LIMITER_CEILING = 0.98
_NORMALIZE_TARGET = 0.89


def auto_chain(analysis):
    """Deterministic mastering chain from measurements.

    Returns (effects, reasons): effects is a list of
    core.edit.apply_chain dicts; reasons names the measurement
    behind each one. Empty effects + a reason saying so when the
    track measures clean -- silence about problems is not allowed,
    and neither is inventing them.
    """
    if not isinstance(analysis, AnalysisResult):
        raise TypeError("auto_chain: expected AnalysisResult, "
                        f"got {type(analysis).__name__}")
    effects, reasons = [], []
    if analysis.clipped_samples > 0:
        effects.append({"type": "limiter", "ceiling": _LIMITER_CEILING,
                        "drive_db": 0.0})
        reasons.append(f"{analysis.clipped_samples} clipped samples: "
                       f"soft limiter at {_LIMITER_CEILING}")
    elif analysis.rms > 0 and analysis.loudness_rms_db > _HOT_RMS_DB:
        effects.append({"type": "limiter", "ceiling": _LIMITER_CEILING,
                        "drive_db": 0.0})
        reasons.append(f"hot master ({analysis.loudness_rms_db:.1f} dB RMS): "
                       "limiter for inter-sample safety")
    if analysis.rms > 0 and analysis.loudness_rms_db < _FAINT_RMS_DB:
        effects.append({"type": "normalize", "target": _NORMALIZE_TARGET})
        reasons.append(f"faint ({analysis.loudness_rms_db:.1f} dB RMS): "
                       f"normalize to {_NORMALIZE_TARGET}")
    if abs(analysis.dc_offset) > _DC_OFFSET_LIMIT:
        effects.append({"type": "highpass", "freq": 20})
        reasons.append(f"DC offset {analysis.dc_offset:+.4f}: "
                       "20 Hz highpass")
    if not effects:
        reasons.append("measures clean: no level problems found, "
                       "chain left empty on purpose")
    return effects, reasons


def build_advice_prompt(analysis, goal):
    """The exact prompt advise() sends (pure function: testable offline)."""
    return (
        "You are a mastering engineer advising a producer. "
        "Below are MEASURED descriptors of their track (RMS loudness, "
        "not LUFS). Their goal: " + goal.strip() + "\n\n"
        + format_report(analysis) + "\n\n"
        "Give concrete, numbered mastering moves: which processor, "
        "which parameter values, and why -- grounded in the numbers "
        "above. If a measurement contradicts the goal, say so plainly. "
        "Do not invent measurements. Keep it under 300 words."
    )


def advise(analysis, goal, api_key=None, model=None):
    """Ask Gemini for mastering advice. Returns GeminiResult.

    Raises GeminiKeyMissing when no key is available -- this is a
    cloud feature and says so loudly instead of degrading into a
    fake local answer.
    """
    from resonance.ai.gemini import DEFAULT_MODEL
    client = GeminiClient(api_key=api_key,
                          model=model or DEFAULT_MODEL)
    prompt = build_advice_prompt(analysis, goal)
    return client.generate(prompt, think_budget=4096, code_execution=True)
