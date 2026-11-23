from __future__ import annotations

import pytest

from biocog import pipeline
from biocog.config import load_config
from biocog.datasets import load_fixture

TINY = {"models": {"n_members": 1, "hidden": 8, "epochs": 2, "patience": 2},
        "federated": {"n_clients": 3, "rounds": 1, "local_epochs": 1}}


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tiny_cfg():
    return load_config(overrides=TINY)


@pytest.fixture(scope="session")
def fixture_tables():
    return load_fixture()


@pytest.fixture(scope="session")
def bundle(tiny_cfg, fixture_tables):
    stays, events = fixture_tables
    return pipeline.build_bundle(stays, events, tiny_cfg)


@pytest.fixture(scope="session")
def tiny_proposed(bundle):
    return pipeline.fit_model("proposed", bundle)
