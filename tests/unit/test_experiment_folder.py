"""ensure_workspace_parent_dir creates the MLflow experiment's parent folder through the Workspace API,
and falls back to os.makedirs on /Workspace when the API is not available."""
import os
import sys
import types

import pytest

from mmf_sa.Forecaster import ensure_workspace_parent_dir


class _Workspace:
    created = []

    def mkdirs(self, path):
        type(self).created.append(path)


class _Client:
    def __init__(self):
        self.workspace = _Workspace()


@pytest.fixture
def fake_sdk(monkeypatch):
    _Workspace.created = []
    sdk = types.ModuleType("databricks.sdk")
    sdk.WorkspaceClient = _Client
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    return _Workspace


@pytest.mark.parametrize("path, parent", [
    ("/Users/me@x.com/mmf/exp", "/Users/me@x.com/mmf"),
    ("/Workspace/Users/me@x.com/mmf/exp", "/Users/me@x.com/mmf"),
])
def test_creates_parent_with_workspace_api(fake_sdk, path, parent):
    ensure_workspace_parent_dir(path)
    assert fake_sdk.created == [parent]


def test_top_level_experiment_needs_no_folder(fake_sdk):
    ensure_workspace_parent_dir("/exp")
    assert fake_sdk.created == []


def test_falls_back_to_makedirs(monkeypatch):
    class _Failing:
        def __init__(self):
            raise ValueError("no credentials")

    sdk = types.ModuleType("databricks.sdk")
    sdk.WorkspaceClient = _Failing
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    made = []
    monkeypatch.setattr(os, "makedirs", lambda p, exist_ok=False: made.append(p))
    ensure_workspace_parent_dir("/Users/me/mmf/exp")
    assert made == ["/Workspace/Users/me/mmf"]
