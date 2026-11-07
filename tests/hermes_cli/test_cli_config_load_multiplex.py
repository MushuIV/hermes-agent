"""Unit tests verifying that _mirror_config_to_env respects multiplexed / routed profile environments."""

import os
import pytest
from hermes_cli.cli_config_load import _mirror_config_to_env
from agent.secret_scope import set_multiplex_active


def test_mirror_config_to_env_preserves_multiplex_environ(monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", "/original/cwd")
    monkeypatch.setenv("BROWSER_INACTIVITY_TIMEOUT", "60")
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")

    set_multiplex_active(True)
    try:
        defaults = {
            "terminal": {"backend": "local", "cwd": "/new/profile/cwd"},
            "browser": {"inactivity_timeout": 120},
            "security": {"redact_secrets": True},
        }
        _mirror_config_to_env(defaults, _file_has_terminal_config=True)

        # Under multiplexing, the process-level environment must NOT be overwritten
        assert os.environ["TERMINAL_CWD"] == "/original/cwd"
        assert os.environ["BROWSER_INACTIVITY_TIMEOUT"] == "60"
        assert os.environ["HERMES_REDACT_SECRETS"] == "false"
    finally:
        set_multiplex_active(False)
