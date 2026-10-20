"""检查基于 CSV 的学习仪表盘在不同输出状态下的行为。"""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from UI import learning_charts


def _metrics(phase: str, *, profit: float = 10.0, episode: int = 2) -> dict:
    return {
        "run": 1, "phase": phase, "episode": episode,
        "total_reward": profit / 10,
        "mean_reward_per_hour": profit / 20,
        "total_profit_eur": profit,
        "accepted_energy_mwh": 5.0,
        "minimum_segment_average_bid_eur_per_mwh": 30.0,
        "flexible_segment_average_bid_eur_per_mwh": 40.0,
        "minimum_segment_acceptance_ratio": 0.5,
        "flexible_segment_acceptance_ratio": 0.25,
    }


class LearningDashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name)

    def _write_learning_run(self):
        pd.DataFrame([
            {"delivery_start": "2020-01-01 00:00", "available_power_mw": 10,
             "clearing_price_eur_per_mwh": 35, "profit_eur": 5, "regret_eur": 1, "reward": 0.5},
            {"delivery_start": "2020-01-01 01:00", "available_power_mw": 10,
             "clearing_price_eur_per_mwh": 36, "profit_eur": 6, "regret_eur": 2, "reward": 0.6},
        ]).to_csv(self.run_dir / "learning_results.csv", index=False)
        pd.DataFrame([
            {"delivery_start": time, "offer_segment": segment,
             "bid_price_eur_per_mwh": bid, "marginal_cost_eur_per_mwh": 20,
             "accepted_power_mw": power}
            for time in ("2020-01-01 00:00", "2020-01-01 01:00")
            for segment, bid, power in (
                ("learning_minimum", 30, 2), ("learning_flexible", 40, 3)
            )
        ]).to_csv(self.run_dir / "offer_results.csv", index=False)

    def test_builds_all_three_learning_charts_from_csv(self):
        self._write_learning_run()
        pd.DataFrame([
            _metrics("initial_experience", episode=1),
            _metrics("train"), _metrics("validation"),
            _metrics("validation_actor"), _metrics("validation_baseline", profit=0),
        ]).to_csv(self.run_dir / "learning_metrics.csv", index=False)

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertEqual(errors, {})
        self.assertEqual(set(figures), {
            "learning_unit", "learning_curves", "learning_validation_comparison"
        })
        self.assertEqual(
            [kind for kind, _, _ in learning_charts.learning_items(self.run_dir, figures, errors)],
            ["figure", "figure", "figure"],
        )
        training = next(trace for trace in figures["learning_curves"].data
                        if trace.name == "Run 1 training")
        self.assertEqual(list(training.x), [1, 2])
        self.assertEqual(len([trace for trace in figures["learning_validation_comparison"].data
                              if trace.type == "bar"]), 16)

    def test_leftover_metrics_do_not_create_learning_charts(self):
        pd.DataFrame([_metrics("validation_actor"), _metrics("validation_baseline")]).to_csv(
            self.run_dir / "learning_metrics.csv", index=False
        )

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertEqual(figures, {})
        self.assertEqual(errors, {})
        self.assertEqual(learning_charts.learning_items(self.run_dir, figures, errors), [])

    def test_newest_metrics_file_wins_for_the_same_period(self):
        self._write_learning_run()
        old = self.run_dir / "learning_metrics.csv"
        new = self.run_dir / "learning_evaluation_metrics.csv"
        pd.DataFrame([_metrics("validation_actor", profit=10),
                      _metrics("validation_baseline", profit=0)]).to_csv(old, index=False)
        pd.DataFrame([_metrics("validation_actor", profit=20),
                      _metrics("validation_baseline", profit=0)]).to_csv(new, index=False)
        os.utime(old, ns=(1_000_000_000, 1_000_000_000))
        os.utime(new, ns=(2_000_000_000, 2_000_000_000))

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertEqual(errors, {})
        self.assertEqual(figures["learning_validation_comparison"].data[0].y[0], 20)
        self.assertIn("learning_evaluation_metrics.csv",
                      figures["learning_validation_comparison"].layout.title.text)

    def test_png_fallback_requires_current_output(self):
        png = self.run_dir / "plots" / "learning_unit.png"
        png.parent.mkdir()
        png.write_bytes(b"png")
        market = self.run_dir / "market_results.csv"
        market.write_text("result", encoding="utf-8")
        os.utime(market, ns=(1_000_000_000, 1_000_000_000))
        os.utime(png, ns=(2_000_000_000, 2_000_000_000))

        self.assertEqual(
            learning_charts.learning_items(self.run_dir, {}, {}),
            [("image", "learning_unit", png)],
        )
        os.utime(market, ns=(3_000_000_000, 3_000_000_000))
        self.assertEqual(learning_charts.learning_items(self.run_dir, {}, {}), [])

    def test_bad_unit_data_does_not_hide_other_learning_charts(self):
        self._write_learning_run()
        pd.DataFrame([_metrics("validation_actor"),
                      _metrics("validation_baseline", profit=0)]).to_csv(
            self.run_dir / "learning_metrics.csv", index=False
        )
        offers = pd.read_csv(self.run_dir / "offer_results.csv")
        offers.drop(columns="offer_segment").to_csv(self.run_dir / "offer_results.csv", index=False)

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertIn("learning_unit", errors)
        self.assertIn("learning_validation_comparison", figures)
        self.assertEqual(
            learning_charts.learning_items(self.run_dir, figures, errors)[0][0], "error"
        )

    def test_test_metrics_produce_test_comparison(self):
        self._write_learning_run()
        pd.DataFrame([_metrics("test_actor"), _metrics("test_baseline", profit=0)]).to_csv(
            self.run_dir / "learning_test_metrics.csv", index=False
        )

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertEqual(errors, {})
        self.assertIsInstance(figures["learning_test_comparison"], go.Figure)


if __name__ == "__main__":
    unittest.main()
