"""Shared constants for tests (imported by name because tests/ is on sys.path via rootdir)."""

TINY_OVERRIDES = {"models": {"n_members": 1, "hidden": 8, "epochs": 2, "patience": 2},
                  "federated": {"n_clients": 3, "rounds": 1, "local_epochs": 1}}
