# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""Sample plugin for the RESONANCE demo: the echo plugin.

Hypothesis: the smallest useful plugin is one hook that logs and one
  hook that transforms, so the demo can show BOTH the manager's log
  stream and a real data pass-through.
Method:     declare PLUGIN metadata with api_version "1.0" (must match
  resonance.API_VERSION exactly -- the manager refuses anything else),
  define register(api), and use @api.on to answer the "ping" hook and
  the "transform" hook.
Observation: api.on is the same decorator the fixture plugins use in
  tests/test_plugins.py; fire("transform", x) returns a list with one
  entry per registered handler.
Result:     a plugin the demo loads from demo/plugins/ that echoes
  "pong:<message>" and doubles numbers -- audible in the log stream
  and in the transform result.

Honesty: this plugin touches no audio pipeline -- it is a harness
demonstration, named as such.
"""

PLUGIN = {
    "name": "echo",
    "version": "0.1.0",
    "api_version": "1.0",
    "description": "demo echo plugin: answers ping, doubles transform input",
}


def register(api):
    """Entry point: the manager calls this with a PluginAPI."""

    @api.on("ping")
    def pong(message=""):
        # The log line is the visible proof the hook fired -- the demo
        # prints the manager's log stream after firing.
        api.log(f"echo heard ping: {message!r}")
        return f"pong:{message}"

    @api.on("transform")
    def double(value):
        # A real pass-through transform: the demo asserts the returned
        # value equals 2 * the input, proving hooks carry data both ways.
        api.log(f"echo doubling {value!r}")
        return value * 2
