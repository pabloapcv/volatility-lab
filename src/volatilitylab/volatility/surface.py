"""
Volatility Surface Construction and Analysis

Builds implied volatility surfaces from option quotes with:
- Smile/skew visualization by strike or moneyness
- Term structure analysis
- Clear separation of observed vs interpolated points
- Sparse data and extrapolation warnings
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate

from volatilitylab.volatility.implied_vol import implied_volatility_batch


def _days_to_years(expiry_days: float) -> float:
    """Convert calendar-day expiry to year fraction."""
    return float(expiry_days) / 365.0


def _tte_years_and_expiry_days(row, has_tte: bool, has_exp: bool) -> tuple[float, float]:
    """
    Return (time_to_expiry_years, expiry_days).

    Surface coordinates are always calendar days so the 3D plot axis matches
    the 'Days to Expiry' label. time_to_expiry stays in years for pricing.
    """
    if has_tte:
        tte = float(row["time_to_expiry"])
        # Defensive: some feeds store TTE already in days (> 3 years is unlikely
        # for listed equity options in this app).
        if tte > 3.0:
            return tte / 365.0, tte
        return tte, tte * 365.0

    if has_exp:
        exp_date = pd.to_datetime(row["expiration"])
        now = pd.Timestamp.now()
        days = float((exp_date - now).days)
        days = max(days, 0.0)
        return days / 365.0, days

    raise ValueError("Need either 'time_to_expiry' or 'expiration'")


def _tte_years_and_expiry_days_vector(
    chain: pd.DataFrame, has_tte: bool, has_exp: bool
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Vectorized (time_to_expiry_years, expiry_days) for a chain DataFrame."""
    if has_tte:
        tte = chain["time_to_expiry"].to_numpy(dtype=np.float64, copy=False)
        stored_as_days = tte > 3.0
        years = np.where(stored_as_days, tte / 365.0, tte)
        days = np.where(stored_as_days, tte, tte * 365.0)
        return years.astype(np.float64, copy=False), days.astype(np.float64, copy=False)

    if has_exp:
        exp_date = pd.to_datetime(chain["expiration"])
        days = (exp_date - pd.Timestamp.now()).dt.days.to_numpy(dtype=np.float64)
        days = np.maximum(days, 0.0)
        return days / 365.0, days

    raise ValueError("Need either 'time_to_expiry' or 'expiration'")


class DataQuality(StrEnum):
    """Data quality classification."""

    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    SPARSE = "sparse"


class SurfacePoint(NamedTuple):
    """Single point on the volatility surface."""

    strike: float
    expiry: float
    iv: float
    quality: DataQuality
    moneyness: float  # K/S or log(K/F)
    bid_iv: float | None = None
    ask_iv: float | None = None


class SmileSlice(NamedTuple):
    """Volatility smile at a single expiry."""

    expiry: float
    time_to_expiry: float
    strikes: NDArray[np.floating]
    ivs: NDArray[np.floating]
    moneyness: NDArray[np.floating]
    qualities: list[DataQuality]
    atm_iv: float | None
    skew: float | None  # dIV/dK at ATM
    convexity: float | None  # d²IV/dK² at ATM


class TermStructurePoint(NamedTuple):
    """Point on the ATM term structure."""

    expiry: float
    time_to_expiry: float
    atm_iv: float
    atm_strike: float
    quality: DataQuality


@dataclass
class SurfaceMetadata:
    """Metadata about surface construction."""

    underlying: str
    spot: float
    observed_at: str
    risk_free_rate: float
    dividend_yield: float
    n_observed_points: int
    n_interpolated_points: int
    n_extrapolated_points: int
    sparse_regions: list[str]
    warnings: list[str]


@dataclass
class VolatilitySurface:
    """
    Implied volatility surface with interpolation and analysis.

    Clearly separates observed points from interpolation and flags
    sparse/extrapolated regions.

    Example
    -------
    >>> surface = VolatilitySurface.from_option_chain(
    ...     chain_df, spot=500, risk_free_rate=0.05
    ... )
    >>> smile = surface.get_smile(expiry_days=30)
    >>> term_structure = surface.get_term_structure()
    """

    points: list[SurfacePoint]
    metadata: SurfaceMetadata
    _interpolator: interpolate.RBFInterpolator | None = field(default=None, repr=False)

    @classmethod
    def from_option_chain(
        cls,
        chain: pd.DataFrame,
        spot: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        underlying: str = "UNKNOWN",
        observed_at: str = "",
        min_tte: float = 1 / 365,  # Minimum 1 day to expiry
        iv_bounds: tuple[float, float] = (0.01, 3.0),
    ) -> "VolatilitySurface":
        """
        Construct surface from an options chain DataFrame.

        Expected columns:
            strike, expiration (or time_to_expiry), option_type,
            bid, ask (or mid_price)
        """

        points: list[SurfacePoint] = []
        warnings: list[str] = []

        # Ensure we have required columns
        required = ["strike", "option_type"]
        has_tte = "time_to_expiry" in chain.columns
        has_exp = "expiration" in chain.columns
        has_bid_ask = "bid" in chain.columns and "ask" in chain.columns
        has_mid = "mid_price" in chain.columns

        if not all(col in chain.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")
        if not (has_tte or has_exp):
            raise ValueError("Need either 'time_to_expiry' or 'expiration' column")
        if not (has_bid_ask or has_mid):
            raise ValueError("Need either 'bid'/'ask' or 'mid_price' columns")

        n = len(chain)
        if n == 0:
            metadata = SurfaceMetadata(
                underlying=underlying,
                spot=spot,
                observed_at=observed_at,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                n_observed_points=0,
                n_interpolated_points=0,
                n_extrapolated_points=0,
                sparse_regions=["entire_surface"],
                warnings=["Very few valid IV points - surface may be unreliable"],
            )
            surface = cls(points=points, metadata=metadata)
            surface._build_interpolator()
            return surface

        strike = chain["strike"].to_numpy(dtype=np.float64, copy=False)
        tte, expiry = _tte_years_and_expiry_days_vector(chain, has_tte=has_tte, has_exp=has_exp)
        opt_type_str = chain["option_type"].astype(str).str.lower()
        is_call = opt_type_str.str.contains("call", regex=False).to_numpy()

        if has_bid_ask:
            bid = pd.to_numeric(chain["bid"], errors="coerce").to_numpy(dtype=np.float64)
            ask = pd.to_numeric(chain["ask"], errors="coerce").to_numpy(dtype=np.float64)
            bid = np.nan_to_num(bid, nan=0.0)
            ask = np.nan_to_num(ask, nan=0.0)
            mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, np.nan)
        else:
            mid = pd.to_numeric(chain["mid_price"], errors="coerce").to_numpy(dtype=np.float64)
            bid = ask = None

        keep = (tte >= min_tte) & np.isfinite(mid) & (mid > 0)
        if not np.any(keep):
            metadata = SurfaceMetadata(
                underlying=underlying,
                spot=spot,
                observed_at=observed_at,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                n_observed_points=0,
                n_interpolated_points=0,
                n_extrapolated_points=0,
                sparse_regions=["entire_surface"],
                warnings=["Very few valid IV points - surface may be unreliable"],
            )
            surface = cls(points=points, metadata=metadata)
            surface._build_interpolator()
            return surface

        mid_iv = implied_volatility_batch(
            mid[keep],
            spot,
            strike[keep],
            tte[keep],
            risk_free_rate,
            is_call[keep],
            dividend_yield,
        )
        in_bounds = np.isfinite(mid_iv) & (mid_iv >= iv_bounds[0]) & (mid_iv <= iv_bounds[1])

        bid_iv_all = None
        ask_iv_all = None
        if bid is not None and ask is not None:
            bid_keep = keep & (bid > 0)
            ask_keep = keep & (ask > 0)
            bid_iv_all = np.full(n, np.nan, dtype=np.float64)
            ask_iv_all = np.full(n, np.nan, dtype=np.float64)
            if np.any(bid_keep):
                bid_iv_all[bid_keep] = implied_volatility_batch(
                    bid[bid_keep],
                    spot,
                    strike[bid_keep],
                    tte[bid_keep],
                    risk_free_rate,
                    is_call[bid_keep],
                    dividend_yield,
                )
            if np.any(ask_keep):
                ask_iv_all[ask_keep] = implied_volatility_batch(
                    ask[ask_keep],
                    spot,
                    strike[ask_keep],
                    tte[ask_keep],
                    risk_free_rate,
                    is_call[ask_keep],
                    dividend_yield,
                )

        kept_idx = np.where(keep)[0]
        tte_k = tte[keep]
        strike_k = strike[keep]
        expiry_k = expiry[keep]
        forward = spot * np.exp((risk_free_rate - dividend_yield) * tte_k)
        moneyness = np.log(strike_k / forward)

        for j, i in enumerate(kept_idx):
            if not in_bounds[j]:
                continue
            b_iv = None
            a_iv = None
            if bid_iv_all is not None:
                val = bid_iv_all[i]
                if np.isfinite(val):
                    b_iv = float(val)
            if ask_iv_all is not None:
                val = ask_iv_all[i]
                if np.isfinite(val):
                    a_iv = float(val)
            points.append(
                SurfacePoint(
                    strike=float(strike_k[j]),
                    expiry=float(expiry_k[j]),
                    iv=float(mid_iv[j]),
                    quality=DataQuality.OBSERVED,
                    moneyness=float(moneyness[j]),
                    bid_iv=b_iv,
                    ask_iv=a_iv,
                )
            )

        # Check for sparse regions
        sparse_regions = []
        if len(points) < 10:
            warnings.append("Very few valid IV points - surface may be unreliable")
            sparse_regions.append("entire_surface")

        metadata = SurfaceMetadata(
            underlying=underlying,
            spot=spot,
            observed_at=observed_at,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            n_observed_points=len(points),
            n_interpolated_points=0,
            n_extrapolated_points=0,
            sparse_regions=sparse_regions,
            warnings=warnings,
        )

        surface = cls(points=points, metadata=metadata)
        surface._build_interpolator()
        return surface

    def _build_interpolator(self) -> None:
        """Build RBF interpolator from observed points."""
        if len(self.points) < 4:
            self._interpolator = None
            return

        # Use moneyness and expiry as coordinates
        coords = np.array([[p.moneyness, p.expiry] for p in self.points])
        ivs = np.array([p.iv for p in self.points])

        try:
            self._interpolator = interpolate.RBFInterpolator(
                coords, ivs, kernel="thin_plate_spline", smoothing=0.001
            )
        except Exception:
            self._interpolator = None

    def get_iv(
        self,
        strike: float,
        expiry: float,
        spot: float | None = None,
    ) -> tuple[float | None, DataQuality]:
        """
        Get IV at a specific strike and expiry.

        Returns
        -------
        tuple[float | None, DataQuality]
            IV and whether it's observed/interpolated/extrapolated
        """
        spot = spot or self.metadata.spot

        # Check for exact observed point
        for p in self.points:
            if abs(p.strike - strike) < 0.01 and abs(p.expiry - expiry) < 0.01:
                return p.iv, DataQuality.OBSERVED

        if self._interpolator is None:
            return None, DataQuality.SPARSE

        # Calculate moneyness
        tte = _days_to_years(expiry)
        forward = spot * np.exp((self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte)
        moneyness = np.log(strike / forward)

        # Check if in extrapolation region
        all_moneyness = [p.moneyness for p in self.points]
        all_expiry = [p.expiry for p in self.points]

        is_extrapolated = (
            moneyness < min(all_moneyness) - 0.05
            or moneyness > max(all_moneyness) + 0.05
            or expiry < min(all_expiry) * 0.9
            or expiry > max(all_expiry) * 1.1
        )

        try:
            iv = float(self._interpolator([[moneyness, expiry]])[0])
            quality = DataQuality.EXTRAPOLATED if is_extrapolated else DataQuality.INTERPOLATED
            return iv, quality
        except Exception:
            return None, DataQuality.SPARSE

    def get_smile(
        self,
        expiry: float,
        n_strikes: int = 20,
        moneyness_range: tuple[float, float] = (-0.3, 0.3),
    ) -> SmileSlice:
        """
        Get volatility smile at a specific expiry.

        Parameters
        ----------
        expiry : float
            Target expiry (days or years depending on data)
        n_strikes : int
            Number of strike points
        moneyness_range : tuple
            Log-moneyness range to sample

        Returns
        -------
        SmileSlice
            Smile data including strikes, IVs, and quality flags
        """
        spot = self.metadata.spot
        tte = _days_to_years(expiry)
        forward = spot * np.exp((self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte)

        # Generate moneyness grid
        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        strikes = forward * np.exp(moneyness)

        ivs = []
        qualities = []

        for k, _m in zip(strikes, moneyness, strict=True):
            iv, quality = self.get_iv(k, expiry)
            ivs.append(iv if iv is not None else np.nan)
            qualities.append(quality)

        ivs_arr = np.array(ivs)

        # Calculate ATM IV and skew
        atm_idx = np.argmin(np.abs(moneyness))
        atm_iv = ivs_arr[atm_idx] if not np.isnan(ivs_arr[atm_idx]) else None

        # Skew: slope of IV vs moneyness at ATM
        skew = None
        convexity = None
        if atm_iv is not None and len(ivs_arr) > 3:
            valid = ~np.isnan(ivs_arr)
            if np.sum(valid) > 3:
                try:
                    coeffs = np.polyfit(moneyness[valid], ivs_arr[valid], 2)
                    skew = coeffs[1]  # Linear term
                    convexity = 2 * coeffs[0]  # Quadratic term
                except Exception:
                    pass

        return SmileSlice(
            expiry=expiry,
            time_to_expiry=tte,
            strikes=strikes,
            ivs=ivs_arr,
            moneyness=moneyness,
            qualities=qualities,
            atm_iv=atm_iv,
            skew=skew,
            convexity=convexity,
        )

    def get_term_structure(
        self,
        moneyness: float = 0.0,
    ) -> list[TermStructurePoint]:
        """
        Get ATM (or fixed moneyness) term structure.

        Parameters
        ----------
        moneyness : float
            Target log-moneyness (0 = ATM)

        Returns
        -------
        list[TermStructurePoint]
            Term structure points sorted by expiry
        """
        # Get unique expiries
        expiries = sorted({p.expiry for p in self.points})

        term_structure = []
        spot = self.metadata.spot

        for expiry in expiries:
            tte = _days_to_years(expiry)
            forward = spot * np.exp(
                (self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte
            )
            strike = forward * np.exp(moneyness)

            iv, quality = self.get_iv(strike, expiry)
            if iv is not None:
                term_structure.append(
                    TermStructurePoint(
                        expiry=expiry,
                        time_to_expiry=tte,
                        atm_iv=iv,
                        atm_strike=strike,
                        quality=quality,
                    )
                )

        return term_structure

    def get_surface_grid(
        self,
        n_strikes: int = 20,
        n_expiries: int = 10,
        moneyness_range: tuple[float, float] = (-0.3, 0.3),
    ) -> dict:
        """
        Generate a grid for 3D surface visualization.

        Returns
        -------
        dict
            Contains moneyness, expiry, iv grids and quality mask
        """
        expiries = sorted({p.expiry for p in self.points})
        if len(expiries) < 2:
            expiries = [min(p.expiry for p in self.points), max(p.expiry for p in self.points)]

        exp_grid = np.linspace(min(expiries), max(expiries), n_expiries)
        m_grid = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)

        M, E = np.meshgrid(m_grid, exp_grid)
        IV = np.zeros_like(M)
        quality_mask = np.zeros_like(M, dtype=int)  # 0=sparse, 1=extrap, 2=interp, 3=observed

        spot = self.metadata.spot

        for i in range(n_expiries):
            for j in range(n_strikes):
                expiry = E[i, j]
                moneyness = M[i, j]
                tte = _days_to_years(expiry)
                forward = spot * np.exp(
                    (self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte
                )
                strike = forward * np.exp(moneyness)

                iv, quality = self.get_iv(strike, expiry)
                if iv is not None:
                    iv = float(np.clip(iv, 0.01, 3.0))
                IV[i, j] = iv if iv is not None else np.nan

                quality_map = {
                    DataQuality.OBSERVED: 3,
                    DataQuality.INTERPOLATED: 2,
                    DataQuality.EXTRAPOLATED: 1,
                    DataQuality.SPARSE: 0,
                }
                quality_mask[i, j] = quality_map[quality]

        return {
            "moneyness": M,
            "expiry": E,
            "iv": IV,
            "quality_mask": quality_mask,
            "observed_points": [
                (p.moneyness, p.expiry, p.iv)
                for p in self.points
                if p.quality == DataQuality.OBSERVED
            ],
        }

    def to_dict(self) -> dict:
        """Convert surface to dictionary for JSON serialization."""
        return {
            "metadata": {
                "underlying": self.metadata.underlying,
                "spot": self.metadata.spot,
                "observed_at": self.metadata.observed_at,
                "risk_free_rate": self.metadata.risk_free_rate,
                "dividend_yield": self.metadata.dividend_yield,
                "n_observed_points": self.metadata.n_observed_points,
                "warnings": self.metadata.warnings,
            },
            "points": [
                {
                    "strike": p.strike,
                    "expiry": p.expiry,
                    "iv": p.iv,
                    "quality": p.quality.value,
                    "moneyness": p.moneyness,
                    "bid_iv": p.bid_iv,
                    "ask_iv": p.ask_iv,
                }
                for p in self.points
            ],
        }
