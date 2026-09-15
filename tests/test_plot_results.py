"""Tests for ablation results plotting and aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.plot_results import load_summary, write_csv, load_individual_results


@pytest.fixture
def ablation_dir(tmp_path):
    """Create a mock ablation directory with eval results."""
    conditions = {
        "baseline": {"success_rate": 0.40, "mean_spl": 0.30, "mean_return": 2.5, "mean_length": 120.0, "episodes": 50},
        "freq_r16": {"success_rate": 0.55, "mean_spl": 0.42, "mean_return": 3.1, "mean_length": 100.0, "episodes": 50},
    }

    summary = {"num_steps": 5000, "eval_episodes": 50, "seed": 0, "conditions": conditions}
    (tmp_path / "ablation_summary.json").write_text(json.dumps(summary))

    for name, res in conditions.items():
        cond_dir = tmp_path / name
        cond_dir.mkdir()
        (cond_dir / "eval_results.json").write_text(json.dumps(res))

    return tmp_path


class TestLoadSummary:
    def test_loads_conditions(self, ablation_dir):
        summary = load_summary(ablation_dir / "ablation_summary.json")
        assert "conditions" in summary
        assert "baseline" in summary["conditions"]
        assert "freq_r16" in summary["conditions"]

    def test_condition_has_metrics(self, ablation_dir):
        summary = load_summary(ablation_dir / "ablation_summary.json")
        baseline = summary["conditions"]["baseline"]
        assert baseline["success_rate"] == pytest.approx(0.40)
        assert baseline["mean_spl"] == pytest.approx(0.30)


class TestLoadIndividualResults:
    def test_finds_conditions(self, ablation_dir):
        results = load_individual_results(ablation_dir)
        assert "baseline" in results
        assert "freq_r16" in results

    def test_ignores_files(self, ablation_dir):
        results = load_individual_results(ablation_dir)
        assert "ablation_summary" not in results


class TestWriteCSV:
    def test_creates_csv(self, ablation_dir, tmp_path):
        results = {
            "baseline": {"success_rate": 0.4, "mean_spl": 0.3, "mean_return": 2.5, "mean_length": 120.0, "episodes": 50},
        }
        csv_path = tmp_path / "out.csv"
        write_csv(results, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "baseline" in content
        assert "success_rate" in content
