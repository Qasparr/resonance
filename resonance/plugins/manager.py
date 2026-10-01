# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.plugins.manager -- discovery, lifecycle, and hook registry.

HYPOTHESIS
    A generative audio engine earns its keep when strangers can extend
    it without forking it: drop a .py file in a plugin directory and the
    engine discovers it, loads it, routes hooks to it, and can unload
    or reload it at runtime. The dangerous half of that bargain is
    version skew -- a plugin written against a different engine API
    silently misbehaving -- so the API contract must be checked loudly
    at load time, and a failed load must never leave the plugin
    half-registered.

METHOD
    1. Discovery: scan a directory for *.py files (skipping __init__.py
       and _private modules). Each file is one plugin; its stem is its
       name. Mirrors the proven bleat pattern: discover() populates a
       registry, load(name) activates one entry.
    2. Load: exec the file in a fresh module namespace
       (resonance_plugin_<name>), read its PLUGIN metadata dict, check
       api_version against resonance.API_VERSION with EXACT string
       equality, then call its register(api) entry point (falling back
       to bleat-style setup(api)). Any failure -- syntax error, missing
       metadata, API mismatch, raising entry point -- raises
       PluginAPIError with the filename in the message, and the plugin
       is removed from sys.modules and never added to the loaded set.
       Never half-loads: the loud check happens BEFORE registration.
    3. Hooks: plugins receive an api object exposing on(hook_name), a
       decorator that binds a handler to a named hook with the plugin
       recorded as owner. manager.fire(hook_name, *args, **kwargs)
       calls handlers in registration order and collects their return
       values into a list. Unload removes exactly the hooks owned by
       the departing plugin -- no ghost bindings.
    4. Unload/reload: unload() tears down hooks, the loaded record, and
       the sys.modules entry. reload() is unload + load, so editing the
       .py file on disk and reloading picks up the new code.

OBSERVATION
    The API_VERSION contract lives on resonance.API_VERSION (builder A's
    resonance/__init__.py owns it; the contract fixes its value at
    "1.0"). This module defers to that canonical value when present and
    falls back to the contract value "1.0" until builder A lands it --
    either way the comparison is exact, never fuzzy.

RESULT
    PluginManager(plugin_dir=None) with discover/load/unload/reload,
    on/fire hook registry, info/loaded introspection, and
    PluginAPIError for every loud failure.
"""

import re
import sys
from pathlib import Path

try:
    # Builder A owns resonance/__init__.py and the canonical constant.
    # When it lands, this defers to it; until then the contract value
    # "1.0" stands in. Either way, the load-time check is EXACT.
    from resonance import API_VERSION
except Exception:  # ImportError / AttributeError: builder A not landed yet
    API_VERSION = "1.0"

# Module names for exec'd plugins. Sanitized (non-word chars -> "_") so
# a file like "my-plugin.py" still yields a valid module name, and
# prefixed so plugin code can never shadow a real resonance module.
_MODULE_PREFIX = "resonance_plugin_"


class PluginAPIError(Exception):
    """Loud plugin failure: discovery, metadata, API mismatch, load,
    unload, or reload. The message always names the plugin file."""


def _module_name(plugin_name):
    """Map a plugin name to its private module namespace."""
    safe = re.sub(r"\W", "_", plugin_name)
    return f"{_MODULE_PREFIX}{safe}"


def _validate_meta(meta, path):
    """Check the PLUGIN metadata dict; return a clean dict or raise.

    Required keys (the contract): name, version, api_version,
    description. Returns exactly those four keys -- nothing the plugin
    smuggled in extra survives into the registry.
    """
    if not isinstance(meta, dict):
        raise PluginAPIError(
            f"plugin {path}: PLUGIN metadata must be a dict, "
            f"got {type(meta).__name__}")
    missing = [k for k in ("name", "version", "api_version", "description")
               if k not in meta]
    if missing:
        raise PluginAPIError(
            f"plugin {path}: PLUGIN metadata missing keys: "
            f"{', '.join(missing)}")
    return {k: str(meta[k]) for k in
            ("name", "version", "api_version", "description")}


class PluginAPI:
    """The api object handed to a plugin's register(api)/setup(api).

    Exposes:
      api.on(hook_name) -- decorator binding a handler to a hook; the
          plugin's name is recorded as the handler's owner so unload()
          can remove exactly its bindings.
      api.log(message)  -- namespaced log line via the manager.
    """

    def __init__(self, manager, plugin_name):
        self._manager = manager
        self._plugin_name = plugin_name

    def on(self, hook_name):
        """Decorator: register the function as a handler for hook_name."""
        def decorator(fn):
            self._manager._add_hook(self._plugin_name, hook_name, fn)
            return fn
        return decorator

    def log(self, message):
        """Log a namespaced line: [plugin-name] message."""
        self._manager._log(self._plugin_name, message)


class PluginManager:
    """Discovers, loads, unloads, reloads plugins; owns the hook registry.

    plugin_dir -- default directory scanned by discover() when no
        explicit directory is passed. May be None; then every discover
        needs an explicit path.
    """

    def __init__(self, plugin_dir=None):
        self._default_dir = plugin_dir
        # name -> Path of the discovered .py file (discovery registry)
        self._discovered = {}
        # name -> {"meta": {...}, "path": Path, "module": module}
        self._loaded = {}
        # hook_name -> [(owner_name_or_None, handler_fn), ...]
        # in registration order; fire() walks this list front to back.
        self._hooks = {}
        self._log_lines = []

    # -- logging ------------------------------------------------------
    def _log(self, plugin_name, message):
        line = f"[{plugin_name}] {message}"
        self._log_lines.append(line)

    def log_lines(self):
        """All namespaced log lines recorded so far (a copy)."""
        return list(self._log_lines)

    # -- discovery ----------------------------------------------------
    def discover(self, plugin_dir=None):
        """Scan a directory for plugin files; return sorted plugin names.

        A plugin file is any *.py directly inside the directory except
        __init__.py and files starting with "_". The file's stem is the
        plugin name. Discovery only records paths -- it executes nothing.
        """
        directory = Path(plugin_dir or self._default_dir or "")
        if not directory.is_dir():
            raise PluginAPIError(
                f"plugin discovery: not a directory: {directory}")
        found = {}
        for path in sorted(directory.glob("*.py")):
            stem = path.stem
            if stem == "__init__" or stem.startswith("_"):
                continue
            found[stem] = path
        self._discovered.update(found)
        return sorted(found)

    # -- the hook registry --------------------------------------------
    def on(self, hook_name):
        """Manager-level decorator: bind a handler owned by no plugin.

        (Plugins use api.on(), which tags ownership so unload() can
        clean up. Manager-level handlers are never unloaded.)
        """
        def decorator(fn):
            self._add_hook(None, hook_name, fn)
            return fn
        return decorator

    def _add_hook(self, owner, hook_name, fn):
        if not callable(fn):
            raise PluginAPIError(
                f"hook {hook_name!r}: handler is not callable")
        self._hooks.setdefault(hook_name, []).append((owner, fn))

    def fire(self, hook_name, *args, **kwargs):
        """Call every handler bound to hook_name, in registration order,
        collecting each return value into a list. Handlers receive the
        same *args/**kwargs. A raising handler aborts the fire and its
        exception propagates to the caller -- loud, like everything
        else here; silent hook failures hide bugs."""
        results = []
        for _owner, fn in list(self._hooks.get(hook_name, [])):
            results.append(fn(*args, **kwargs))
        return results

    def hooks(self, hook_name=None):
        """Introspection: {hook_name: [owner, ...]} or owners of one hook."""
        if hook_name is not None:
            return [owner for owner, _fn
                    in self._hooks.get(hook_name, [])]
        return {name: [owner for owner, _fn in handlers]
                for name, handlers in self._hooks.items()}

    # -- lifecycle ----------------------------------------------------
    def load(self, name):
        """Load a discovered plugin; return its metadata dict.

        Loud failures (PluginAPIError, filename always in the message):
          - name was never discovered,
          - name is already loaded (double load),
          - the file has a syntax error,
          - PLUGIN metadata is missing/invalid,
          - api_version != resonance.API_VERSION (EXACT match required),
          - no register(api)/setup(api) entry point,
          - the entry point raises.
        On ANY failure the plugin is NOT in the loaded set and its
        module is dropped from sys.modules: never half-loads.
        """
        if name in self._loaded:
            raise PluginAPIError(f"plugin {name!r}: already loaded")
        path = self._discovered.get(name)
        if path is None:
            raise PluginAPIError(
                f"plugin {name!r}: not discovered; call discover() first")
        path = Path(path)
        mod_name = _module_name(name)

        # Fresh namespace per load: stale sys.modules entries (from a
        # previous unload or a failed load) must not leak through.
        sys.modules.pop(mod_name, None)
        try:
            # Read + compile the source ourselves instead of going
            # through importlib's SourceFileLoader. Reason: the loader
            # caches bytecode in __pycache__ keyed on (mtime, size), and
            # a plugin edited twice within one timestamp tick with the
            # same file size would silently re-run the STALE bytecode --
            # exactly the "reload picks up a modified file" case. The
            # stdlib `types` module plus compile() has no such cache.
            from types import ModuleType
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PluginAPIError(
                    f"plugin {path}: cannot read file: {exc}")
            try:
                code = compile(source, str(path), "exec")
            except SyntaxError as exc:
                # The contract's loud case: filename IN the message.
                raise PluginAPIError(
                    f"plugin {path}: syntax error: {exc}")
            module = ModuleType(mod_name)
            module.__file__ = str(path)
            sys.modules[mod_name] = module
            try:
                exec(code, module.__dict__)
            except Exception as exc:
                raise PluginAPIError(
                    f"plugin {path}: failed during exec: "
                    f"{type(exc).__name__}: {exc}")
        except PluginAPIError:
            sys.modules.pop(mod_name, None)  # never half-load
            raise

        # -- metadata BEFORE registration: the API check gates entry --
        try:
            meta = _validate_meta(getattr(module, "PLUGIN", None), path)
            declared = meta["api_version"]
            if declared != str(API_VERSION):
                raise PluginAPIError(
                    f"plugin {path}: api_version {declared!r} does not "
                    f"match resonance API_VERSION {str(API_VERSION)!r}; "
                    f"refusing to load")
            entry = getattr(module, "register", None)
            if entry is None:
                # Bleat-compatible fallback: setup(api).
                entry = getattr(module, "setup", None)
            if entry is None or not callable(entry):
                raise PluginAPIError(
                    f"plugin {path}: no register(api) or setup(api) "
                    f"entry point found")
            try:
                entry(PluginAPI(self, name))
            except Exception as exc:
                raise PluginAPIError(
                    f"plugin {path}: entry point raised "
                    f"{type(exc).__name__}: {exc}")
        except PluginAPIError:
            sys.modules.pop(mod_name, None)  # never half-load
            raise

        self._loaded[name] = {"meta": meta, "path": path, "module": module}
        self._log(name, f"loaded (api {meta['api_version']})")
        return dict(meta)

    def unload(self, name):
        """Unload a plugin: remove its hooks, drop its module, forget it.

        Raises PluginAPIError if the plugin is not loaded. After unload,
        fire() no longer reaches its handlers -- no ghost bindings.
        """
        record = self._loaded.get(name)
        if record is None:
            raise PluginAPIError(f"plugin {name!r}: not loaded")
        # Remove exactly this plugin's hooks; other owners are untouched.
        for hook_name in list(self._hooks):
            kept = [(owner, fn) for owner, fn in self._hooks[hook_name]
                    if owner != name]
            if kept:
                self._hooks[hook_name] = kept
            else:
                del self._hooks[hook_name]
        sys.modules.pop(_module_name(name), None)
        del self._loaded[name]
        self._log(name, "unloaded")

    def reload(self, name):
        """Unload then load: picks up a modified .py file from disk.

        Raises PluginAPIError if the plugin is not loaded, or if the
        re-load fails (in which case the plugin stays unloaded and the
        error names the file -- the caller decides whether to retry).
        Returns the fresh metadata dict.
        """
        if name not in self._loaded:
            raise PluginAPIError(
                f"plugin {name!r}: not loaded; cannot reload")
        self.unload(name)
        return self.load(name)

    # -- introspection --------------------------------------------------
    def loaded(self):
        """Sorted names of currently loaded plugins."""
        return sorted(self._loaded)

    def info(self, name):
        """Metadata dict for a discovered plugin, plus loaded state."""
        path = self._discovered.get(name)
        if path is None:
            raise PluginAPIError(f"plugin {name!r}: not discovered")
        record = self._loaded.get(name)
        info = {"name": name, "path": str(path),
                "loaded": record is not None}
        if record is not None:
            info.update(record["meta"])
        return info
