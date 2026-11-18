"""Shared pytest setup: headless matplotlib backend for every test."""

import matplotlib

matplotlib.use("Agg")
