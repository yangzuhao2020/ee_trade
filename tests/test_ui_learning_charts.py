"""检查基于 CSV 的学习仪表盘在不同输出状态下的行为。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
             "accepted_power_mw": 5, "clearing_price_eur_per_mwh": 35,
             "profit_eur": 5, "regret_eur": 1, "reward": 0.5},
            {"delivery_start": "2020-01-01 01:00", "available_power_mw": 10,
             "accepted_power_mw": 5, "clearing_price_eur_per_mwh": 36,
             "profit_eur": 6, "regret_eur": 2, "reward": 0.6},
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
                        if trace.name == "训练")
        self.assertEqual(list(training.x), [1, 2])
        self.assertEqual(len([trace for trace in figures["learning_validation_comparison"].data
                              if trace.type == "bar"]), 8)

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

    def test_unexpected_errors_propagate(self):
        self._write_learning_run()
        pd.DataFrame([
            _metrics("train"), _metrics("validation"),
            _metrics("validation_actor"), _metrics("validation_baseline"),
        ]).to_csv(self.run_dir / "learning_metrics.csv", index=False)

        for owner, name in (
            (pd, "read_csv"),
            (learning_charts, "learning_unit"),
            (learning_charts, "learning_curves"),
            (learning_charts, "learning_comparison"),
        ):
            with self.subTest(name=name), patch.object(
                owner, name, side_effect=RuntimeError("internal bug")
            ):
                with self.assertRaisesRegex(RuntimeError, "internal bug"):
                    learning_charts.build_learning_charts(self.run_dir)

    def test_incomplete_comparison_reports_error_instead_of_png(self):
        self._write_learning_run()
        (self.run_dir / "plots").mkdir()
        for period, filename in (
            ("train", "learning_evaluation_metrics.csv"),
            ("validation", "learning_metrics.csv"),
            ("test", "learning_test_metrics.csv"),
        ):
            name = f"learning_{period}_comparison"
            (self.run_dir / "plots" / f"{name}.png").write_bytes(b"png")
            for present, missing in (("actor", "baseline"), ("baseline", "actor")):
                with self.subTest(period=period, missing=missing):
                    pd.DataFrame([_metrics(f"{period}_{present}")]).to_csv(
                        self.run_dir / filename, index=False
                    )

                    figures, errors = learning_charts.build_learning_charts(self.run_dir)

                    self.assertNotIn(name, figures)
                    self.assertIn(f"{period}_{missing}", errors[name])
                    self.assertIn(filename, errors[name])
                    item = next(item for item in learning_charts.learning_items(
                        self.run_dir, figures, errors
                    ) if item[1] == name)
                    self.assertEqual(item[0], "error")
            (self.run_dir / filename).unlink()

    def test_newest_incomplete_metrics_do_not_fall_back_to_older_data(self):
        self._write_learning_run()
        old = self.run_dir / "learning_metrics.csv"
        new = self.run_dir / "learning_evaluation_metrics.csv"
        pd.DataFrame([_metrics("validation_actor"),
                      _metrics("validation_baseline")]).to_csv(old, index=False)
        pd.DataFrame([_metrics("validation_actor")]).to_csv(new, index=False)
        os.utime(old, ns=(1_000_000_000, 1_000_000_000))
        os.utime(new, ns=(2_000_000_000, 2_000_000_000))

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertNotIn("learning_validation_comparison", figures)
        self.assertIn(new.name, errors["learning_validation_comparison"])

    def test_unit_power_comes_from_learning_results(self):
        self._write_learning_run()
        results_path = self.run_dir / "learning_results.csv"
        results = pd.read_csv(results_path)
        results["accepted_power_mw"] = [4, 7]
        results.iloc[::-1].to_csv(results_path, index=False)
        offers_path = self.run_dir / "offer_results.csv"
        offers = pd.read_csv(offers_path)
        offers.drop(columns="accepted_power_mw").to_csv(offers_path, index=False)

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertEqual(errors, {})
        power = next(trace for trace in figures["learning_unit"].data
                     if trace.name == "总成交功率")
        self.assertEqual(list(power.y), [4, 7])

    def test_missing_result_power_reports_input_error(self):
        self._write_learning_run()
        path = self.run_dir / "learning_results.csv"
        results = pd.read_csv(path)
        results.drop(columns="accepted_power_mw").to_csv(path, index=False)

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertNotIn("learning_unit", figures)
        self.assertIn("accepted_power_mw", errors["learning_unit"])

    def test_missing_offers_report_error_instead_of_png(self):
        self._write_learning_run()
        (self.run_dir / "offer_results.csv").unlink()
        png = self.run_dir / "plots" / "learning_unit.png"
        png.parent.mkdir()
        png.write_bytes(b"png")

        figures, errors = learning_charts.build_learning_charts(self.run_dir)

        self.assertNotIn("learning_unit", figures)
        self.assertIn("offer_results.csv", errors["learning_unit"])
        self.assertEqual(learning_charts.learning_items(self.run_dir, figures, errors),
                         [("error", "learning_unit", errors["learning_unit"])])

    def test_unreadable_metrics_report_file_error_once_without_png(self):
        self._write_learning_run()
        plots = self.run_dir / "plots"
        plots.mkdir()
        for name in learning_charts.CHART_NAMES[1:]:
            (plots / f"{name}.png").write_bytes(b"png")

        for filename in learning_charts.DATA_FILES[1:]:
            path = self.run_dir / filename
            for content in ("", "phase\nvalidation_actor\n"):
                with self.subTest(filename=filename, content=content):
                    path.write_text(content, encoding="utf-8")

                    figures, errors = learning_charts.build_learning_charts(self.run_dir)

                    self.assertEqual(set(errors), {filename})
                    self.assertEqual(set(figures), {"learning_unit"})
                    self.assertEqual(
                        [(kind, name) for kind, name, _ in learning_charts.learning_items(
                            self.run_dir, figures, errors
                        )],
                        [("figure", "learning_unit")],
                    )
            path.unlink()

    def test_unreadable_evaluation_does_not_replace_newer_data_or_hide_other_charts(self):
        self._write_learning_run()
        training = self.run_dir / "learning_metrics.csv"
        evaluation = self.run_dir / "learning_evaluation_metrics.csv"
        pd.DataFrame([
            _metrics("train"), _metrics("validation"),
            _metrics("validation_actor", profit=10), _metrics("validation_baseline"),
        ]).to_csv(training, index=False)
        pd.DataFrame([_metrics("test_actor"), _metrics("test_baseline")]).to_csv(
            self.run_dir / "learning_test_metrics.csv", index=False
        )
        bad_row = _metrics("validation_actor")
        bad_row["total_profit_eur"] = "invalid"
        pd.DataFrame([bad_row, _metrics("validation_baseline")]).to_csv(
            evaluation, index=False
        )
        os.utime(training, ns=(2_000_000_000, 2_000_000_000))
        for timestamp in (1_000_000_000, 3_000_000_000):
            with self.subTest(evaluation_timestamp=timestamp):
                os.utime(evaluation, ns=(timestamp, timestamp))

                figures, errors = learning_charts.build_learning_charts(self.run_dir)

                self.assertEqual(set(errors), {evaluation.name})
                self.assertTrue({"learning_unit", "learning_curves",
                                 "learning_test_comparison"}.issubset(figures))
                if timestamp < 2_000_000_000:
                    self.assertEqual(figures["learning_validation_comparison"].data[0].y[0], 10)
                else:
                    self.assertNotIn("learning_validation_comparison", figures)


if __name__ == "__main__":
    unittest.main()
