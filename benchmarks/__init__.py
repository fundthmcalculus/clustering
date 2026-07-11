"""Benchmark and profiling harness for tribble-clustering.

This package is a *development* tool (not shipped in the wheel). It measures
wall-clock time and peak resident memory of the core O(n^2) pipeline stages
across dataset sizes, verifies correctness against reference implementations,
and records results as JSON baselines so later optimizations can be compared
against a fixed reference point.
"""
