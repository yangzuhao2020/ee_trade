"""图表 CSV 加载边界：必需文件、无表头文件及合法空表。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from UI import charts


class ChartLoadingContracts(unittest.TestCase):
    HEADER = "delivery_start,delivery_end\n"

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)
        for key in charts.REQUIRED_TABLES:
            (self.run_dir / charts.TABLES[key]).write_text(self.HEADER, encoding="utf-8")

    def test_missing_required_csv_raises(self):
        for key in charts.REQUIRED_TABLES:
            with self.subTest(table=key):
                path = self.run_dir / charts.TABLES[key]
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    charts.load_run(self.run_dir)
                path.write_text(self.HEADER, encoding="utf-8")

    def test_headerless_csv_raises_for_required_and_optional_tables(self):
        for key, filename in charts.TABLES.items():
            with self.subTest(table=key):
                path = self.run_dir / filename
                path.write_text("", encoding="utf-8")
                with self.assertRaises(pd.errors.EmptyDataError):
                    charts.load_run(self.run_dir)
                path.write_text(self.HEADER, encoding="utf-8")

    def test_header_only_tables_keep_required_and_skip_optional(self):
        for filename in charts.TABLES.values():
            (self.run_dir / filename).write_text(self.HEADER, encoding="utf-8")
        tables = charts.load_run(self.run_dir)
        self.assertEqual(set(tables), set(charts.REQUIRED_TABLES))
        self.assertTrue(all(table.empty for table in tables.values()))
        self.assertEqual(charts.build_charts(tables), [])


class HouseholdSavingsColumns(unittest.TestCase):
    def test_unused_payment_column_is_not_required(self):
        household = pd.DataFrame({
            "unit_name": ["home"],
            "delivery_start": pd.to_datetime(["2024-01-01 00:00"]),
            "delivery_end": pd.to_datetime(["2024-01-01 01:00"]),
            "payment_eur": [4.0],
            "baseline_payment_eur": [5.0],
            "actual_saving_eur": [1.0],
            "unmet_electricity_mwh": [0.0],
            "baseline_unmet_electricity_mwh": [0.0],
            "unmet_heat_mwh_th": [0.0],
            "baseline_unmet_heat_mwh_th": [0.0],
            "soc_after": [0.5],
            "baseline_soc_after": [0.5],
        })
        expected = charts.household_savings({"household": household})
        actual = charts.household_savings({
            "household": household.drop(columns="payment_eur"),
        })
        self.assertEqual(json.loads(actual.to_json()), json.loads(expected.to_json()))


if __name__ == "__main__":
    unittest.main()
