# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_plugins.py -- script-style tests for resonance.plugins.

Run:  python3 tests/test_plugins.py      (from the repo root)
   or python3 -m pytest tests/test_plugins.py

Style (matching bleat/trvvth): each test prints "  ok: <name>"; the end
prints "<N> plugins tests passed." Any failure raises immediately -- the
first red line is the diagnosis.

Covers: discovery of .py files, load/unload/reload lifecycle, the hook
registry (on/fire, results collected in registration order), the
API_VERSION contract (exact match; mismatch raises PluginAPIError and
never half-loads), reload picking up modified files, and loud failure
for a syntax-broken plugin with the filename in the message.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.plugins.manager import (
    API_VERSION,
    PluginAPIError,
    PluginManager,
)

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def plugin_source(name, hook="ping", reply="pong", api_version="1.0",
                  description="fixture plugin"):
    """Generate a minimal plugin file's source text."""
    return (
        f'PLUGIN = {{\n'
        f'    "name": {name!r},\n'
        f'    "version": "0.1.0",\n'
        f'    "api_version": {api_version!r},\n'
        f'    "description": {description!r},\n'
        f'}}\n'
        f'\n'
        f'def register(api):\n'
        f'    @api.on({hook!r})\n'
        f'    def handler(*args, **kwargs):\n'
        f'        return {reply!r}\n'
    )


def make_plugin_dir(files):
    """A temp plugin dir holding {filename: source}. Caller owns cleanup
    via the returned TemporaryDirectory."""
    tmp = tempfile.TemporaryDirectory()
    for filename, source in files.items():
        (Path(tmp.name) / filename).write_text(source)
    return tmp


def t_api_version_contract_value():
    # The contract fixes resonance.API_VERSION at "1.0"; the manager
    # defers to builder A's resonance/__init__.py when it lands.
    assert API_VERSION == "1.0", API_VERSION
check("t_api_version_contract_value", t_api_version_contract_value)


def t_discover_finds_py_files():
    tmp = make_plugin_dir({
        "alpha.py": plugin_source("alpha"),
        "beta.py": plugin_source("beta"),
        "_private.py": plugin_source("nope"),
        "__init__.py": "",
        "notes.txt": "not a plugin",
    })
    mgr = PluginManager()
    found = mgr.discover(tmp.name)
    assert found == ["alpha", "beta"], found  # _private + txt skipped
    tmp.cleanup()
check("t_discover_finds_py_files", t_discover_finds_py_files)


def t_discover_rejects_non_directory():
    mgr = PluginManager()
    try:
        mgr.discover("/tmp/definitely_not_a_resonance_dir_xyz")
    except PluginAPIError:
        return
    raise AssertionError("non-directory accepted")
check("t_discover_rejects_non_directory", t_discover_rejects_non_directory)


def t_load_unload_lifecycle():
    tmp = make_plugin_dir({"alpha.py": plugin_source("alpha")})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    meta = mgr.load("alpha")
    assert meta["name"] == "alpha"
    assert meta["version"] == "0.1.0"
    assert meta["api_version"] == "1.0"
    assert meta["description"] == "fixture plugin"
    assert mgr.loaded() == ["alpha"]
    # The hook fires through the registry.
    assert mgr.fire("ping") == ["pong"]
    mgr.unload("alpha")
    assert mgr.loaded() == []
    assert mgr.fire("ping") == []  # no ghost bindings after unload
    assert mgr.hooks("ping") == []
    tmp.cleanup()
check("t_load_unload_lifecycle", t_load_unload_lifecycle)


def t_loads_fixture_plugin_from_tests():
    # The tiny fixture plugin that ships inside tests/ loads for real.
    mgr = PluginManager()
    found = mgr.discover(str(FIXTURES_DIR))
    assert "tiny_plugin" in found
    meta = mgr.load("tiny_plugin")
    assert meta["name"] == "tiny" and meta["api_version"] == "1.0"
    assert mgr.fire("ping") == ["pong:hello"]
    assert any("[tiny_plugin]" in line for line in mgr.log_lines())
    mgr.unload("tiny_plugin")
check("t_loads_fixture_plugin_from_tests", t_loads_fixture_plugin_from_tests)


def t_hook_collects_in_registration_order():
    # Two plugins, one hook: results come back in registration order.
    tmp = make_plugin_dir({
        "aaa_first.py": plugin_source("aaa_first", reply="first"),
        "zzz_second.py": plugin_source("zzz_second", reply="second"),
    })
    mgr = PluginManager()
    mgr.discover(tmp.name)
    mgr.load("aaa_first")
    mgr.load("zzz_second")
    assert mgr.fire("ping") == ["first", "second"]
    # Unloading the first leaves the second's binding intact.
    mgr.unload("aaa_first")
    assert mgr.fire("ping") == ["second"]
    tmp.cleanup()
check("t_hook_collects_in_registration_order",
      t_hook_collects_in_registration_order)


def t_hook_args_pass_through():
    tmp = make_plugin_dir({
        "echo.py": (
            'PLUGIN = {"name": "echo", "version": "0.1.0",\n'
            '          "api_version": "1.0", "description": "echo"}\n'
            'def register(api):\n'
            '    @api.on("sum")\n'
            '    def add(a, b=0):\n'
            '        return a + b\n'
        ),
    })
    mgr = PluginManager()
    mgr.discover(tmp.name)
    mgr.load("echo")
    assert mgr.fire("sum", 2, b=3) == [5]
    assert mgr.fire("unknown_hook") == []  # no handlers: empty, not error
    tmp.cleanup()
check("t_hook_args_pass_through", t_hook_args_pass_through)


def t_api_mismatch_raises_and_never_half_loads():
    tmp = make_plugin_dir({
        "old.py": plugin_source("old", api_version="0.9"),
    })
    mgr = PluginManager()
    mgr.discover(tmp.name)
    try:
        mgr.load("old")
    except PluginAPIError as exc:
        assert "0.9" in str(exc), str(exc)  # the mismatch is named
    else:
        raise AssertionError("api_version 0.9 was accepted")
    # Never half-loads: not in the loaded set, no hooks registered.
    assert mgr.loaded() == []
    assert mgr.fire("ping") == []
    assert mgr.hooks("ping") == []
    tmp.cleanup()
check("t_api_mismatch_raises_and_never_half_loads",
      t_api_mismatch_raises_and_never_half_loads)


def t_reload_picks_up_modified_file():
    tmp = make_plugin_dir({"mod.py": plugin_source("mod", reply="v1")})
    pdir = Path(tmp.name)
    mgr = PluginManager()
    mgr.discover(tmp.name)
    mgr.load("mod")
    assert mgr.fire("ping") == ["v1"]
    # Edit the file on disk, reload, new code answers.
    (pdir / "mod.py").write_text(plugin_source("mod", reply="v2"))
    meta = mgr.reload("mod")
    assert meta["api_version"] == "1.0"
    assert mgr.fire("ping") == ["v2"]
    # Exactly one binding: reload did not stack a second copy.
    assert len(mgr.hooks("ping")) == 1
    tmp.cleanup()
check("t_reload_picks_up_modified_file", t_reload_picks_up_modified_file)


def t_broken_plugin_fails_loudly_with_filename():
    tmp = make_plugin_dir({"broken.py": "def oops(:\n    this is not python\n"})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    try:
        mgr.load("broken")
    except PluginAPIError as exc:
        assert "broken.py" in str(exc), str(exc)  # filename in message
    else:
        raise AssertionError("syntax-broken plugin loaded")
    assert mgr.loaded() == []
    tmp.cleanup()
check("t_broken_plugin_fails_loudly_with_filename",
      t_broken_plugin_fails_loudly_with_filename)


def t_missing_metadata_rejected():
    tmp = make_plugin_dir({"nometa.py": "def register(api):\n    pass\n"})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    try:
        mgr.load("nometa")
    except PluginAPIError as exc:
        assert "nometa.py" in str(exc)
    else:
        raise AssertionError("plugin without PLUGIN metadata loaded")
    assert mgr.loaded() == []
    tmp.cleanup()
check("t_missing_metadata_rejected", t_missing_metadata_rejected)


def t_missing_entry_point_rejected():
    src = ('PLUGIN = {"name": "noentry", "version": "0.1.0",\n'
           '          "api_version": "1.0", "description": "x"}\n')
    tmp = make_plugin_dir({"noentry.py": src})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    try:
        mgr.load("noentry")
    except PluginAPIError:
        pass
    else:
        raise AssertionError("plugin without entry point loaded")
    assert mgr.loaded() == []
    tmp.cleanup()
check("t_missing_entry_point_rejected", t_missing_entry_point_rejected)


def t_entry_point_raising_reported():
    src = ('PLUGIN = {"name": "raiser", "version": "0.1.0",\n'
           '          "api_version": "1.0", "description": "x"}\n'
           'def register(api):\n'
           '    raise RuntimeError("boom")\n')
    tmp = make_plugin_dir({"raiser.py": src})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    try:
        mgr.load("raiser")
    except PluginAPIError as exc:
        assert "boom" in str(exc) and "raiser.py" in str(exc)
    else:
        raise AssertionError("raising entry point loaded")
    assert mgr.loaded() == [], "failed load must not half-register"
    tmp.cleanup()
check("t_entry_point_raising_reported", t_entry_point_raising_reported)


def t_double_load_raises():
    tmp = make_plugin_dir({"alpha.py": plugin_source("alpha")})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    mgr.load("alpha")
    try:
        mgr.load("alpha")
    except PluginAPIError:
        pass
    else:
        raise AssertionError("double load accepted")
    mgr.unload("alpha")
    tmp.cleanup()
check("t_double_load_raises", t_double_load_raises)


def t_unload_unknown_raises():
    mgr = PluginManager()
    try:
        mgr.unload("ghost")
    except PluginAPIError:
        return
    raise AssertionError("unload of unknown plugin accepted")
check("t_unload_unknown_raises", t_unload_unknown_raises)


def t_info_reports_state():
    tmp = make_plugin_dir({"alpha.py": plugin_source("alpha")})
    mgr = PluginManager()
    mgr.discover(tmp.name)
    info = mgr.info("alpha")
    assert info["name"] == "alpha" and info["loaded"] is False
    mgr.load("alpha")
    info = mgr.info("alpha")
    assert info["loaded"] is True and info["version"] == "0.1.0"
    assert info["path"].endswith("alpha.py")
    tmp.cleanup()
check("t_info_reports_state", t_info_reports_state)


print(f"\n{PASSED} plugins tests passed.")
