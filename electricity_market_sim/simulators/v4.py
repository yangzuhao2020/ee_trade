"""Version four industrial demand-side management simulation."""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..errors import InputValidationError
from ..industry import IndustryRollingCoordinator, missing_industrial_price_names
from ..inputs import (
    load_aligned_fuel_price_profiles,
    load_aligned_time_series_profiles,
    load_industrial_units,
)
from ..market_models import DemandBid, MarketClearingResult
from ..models import (
    IndustrialUnit,
    MarketOpening,
    MarketSettings,
    SimulationResult,
)
from .eom import (
    EomExtensionInputs,
    ScheduledProductKey,
    simulate_eom_market,
)


class _IndustryMarketExtension:
    """Connect industrial rolling optimization to the shared EOM event loop."""

    def __init__(
        self,
        input_path: Path,
        settings: MarketSettings,
        units: tuple[IndustrialUnit, ...],
    ) -> None:
        self.input_path = input_path
        self.settings = settings
        self.units = units
        self.profiles: dict[str, dict[datetime, float]] = {}
        self.fuel_price_profiles: dict[datetime, dict[str, float]] = {}
        self.coordinator: IndustryRollingCoordinator | None = None

    def prepare(self, *, requires_market_price_forecast: bool) -> EomExtensionInputs:
        external_forecast_columns = (
            {"electricity_price"} if requires_market_price_forecast else set()
        )
        required_forecasts = tuple(
            sorted(
                external_forecast_columns
                | {unit.forecast_price_column for unit in self.units}
                | {f"{unit.name}_normalized_load_profile" for unit in self.units}
            )
        )
        self.profiles = load_aligned_time_series_profiles(
            self.input_path / "forecasts_df.csv",
            required_forecasts,
            self.settings.product_duration,
        )
        self.fuel_price_profiles = load_aligned_fuel_price_profiles(
            self.input_path / "fuel_prices_df.csv",
            self.settings.product_duration,
        )
        if not self.fuel_price_profiles:
            raise InputValidationError(
                "fuel_prices_df.csv contains no complete products."
            )
        for missing_price in missing_industrial_price_names(self.fuel_price_profiles):
            warnings.warn(
                f"fuel_prices_df.csv is missing {missing_price!r}; industrial "
                "optimization uses the documented default price 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        market_price_forecasts = (
            self.profiles["electricity_price"]
            if requires_market_price_forecast
            else None
        )
        return EomExtensionInputs(
            participant_names=tuple(unit.name for unit in self.units),
            fuel_prices=next(iter(self.fuel_price_profiles.values())),
            fuel_price_profiles=self.fuel_price_profiles,
            market_price_forecasts=market_price_forecasts,
        )

    def initialize(self, openings: tuple[MarketOpening, ...]) -> None:
        product_list = [
            product
            for opening in openings
            for product in opening.products
            if product[1] <= self.settings.end
        ]
        if len(product_list) != len(set(product_list)):
            raise InputValidationError(
                "Industrial rolling optimization requires non-overlapping EOM "
                "openings so each delivery product is committed once."
            )
        self.coordinator = IndustryRollingCoordinator(
            units=self.units,
            all_products=tuple(sorted(product_list)),
            price_forecasts={
                unit.name: self.profiles[unit.forecast_price_column]
                for unit in self.units
            },
            normalized_load_profiles={
                unit.name: self.profiles[f"{unit.name}_normalized_load_profile"]
                for unit in self.units
            },
            fuel_price_profiles=self.fuel_price_profiles,
            maximum_bid_price=self.settings.maximum_bid_price,
        )

    def _require_coordinator(self) -> IndustryRollingCoordinator:
        if self.coordinator is None:
            raise RuntimeError("Industrial market extension is not initialized.")
        return self.coordinator

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[DemandBid, ...]:
        return tuple(
            self._require_coordinator().bids_for_products(products, scheduled_results)
        )

    def record_delivery(self, result: MarketClearingResult) -> None:
        self._require_coordinator().record_delivery(result)

    def finalize(self, result: SimulationResult) -> SimulationResult:
        coordinator = self._require_coordinator()
        coordinator.validate_completion()
        return replace(
            result,
            industry_results=tuple(
                sorted(
                    coordinator.dispatch_results,
                    key=lambda item: (item.delivery_start, item.unit_name),
                )
            ),
            industry_flexibility_results=tuple(
                sorted(
                    coordinator.flexibility_results,
                    key=lambda item: (item.delivery_start, item.unit_name),
                )
            ),
            industry_optimization_windows=tuple(
                sorted(
                    coordinator.window_results,
                    key=lambda item: (item.optimization_start, item.unit_name),
                )
            ),
        )


def _load_and_validate_industrial_units(
    input_path: Path, settings: MarketSettings
) -> tuple[IndustrialUnit, ...]:
    if settings.industrial_dsm_units_file is None:
        raise InputValidationError(
            "Version four requires industrial_dsm_units in config.yaml."
        )
    units = load_industrial_units(input_path / settings.industrial_dsm_units_file)
    if not units:
        raise InputValidationError(
            "The configured industrial_dsm_units file must contain at least one "
            "steel plant."
        )
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError(
            "Version four industrial rolling optimization currently requires "
            "market_mechanism: pay_as_clear."
        )
    for unit in units:
        for field_name, horizon in (
            ("look_ahead_horizon", unit.look_ahead),
            ("commit_horizon", unit.commit_horizon),
            ("rolling_step", unit.rolling_step),
        ):
            ratio = horizon.total_seconds() / settings.product_duration.total_seconds()
            if abs(ratio - round(ratio)) > 1e-9:
                raise InputValidationError(
                    f"Industrial {field_name} for {unit.name!r} must be an "
                    "exact multiple of the EOM product duration."
                )
    return units


def simulate_v4_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
) -> SimulationResult:
    """Calculate one V4 scenario with industrial rolling optimization."""

    units = _load_and_validate_industrial_units(input_path, settings)
    extension = _IndustryMarketExtension(input_path, settings, units)
    return simulate_eom_market(
        input_path,
        settings,
        openings,
        extension=extension,
    )
