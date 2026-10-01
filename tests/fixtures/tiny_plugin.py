# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""Tiny fixture plugin for the resonance plugin-manager tests.

Not shipped as a demo -- it exists so the machinery has something real
to load. Declares api_version "1.0" to match resonance.API_VERSION, and
answers the "ping" hook.
"""

PLUGIN = {
    "name": "tiny",
    "version": "0.1.0",
    "api_version": "1.0",
    "description": "tiny fixture plugin for the plugin-manager tests",
}


def register(api):
    @api.on("ping")
    def pong(*args, **kwargs):
        return "pong:hello"

    api.log("tiny is listening for ping")
