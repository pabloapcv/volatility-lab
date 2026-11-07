"""
Volatility Surface Construction and Analysis

Builds implied volatility surfaces from option quotes with:
- Smile/skew visualization by strike or moneyness
- Term structure analysis
- Clear separation of observed vs interpolated points
- Sparse data and extrapolation warnings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

from volatilitylab.pricing.black_scholes import OptionType
from volatilitylab.volatility.implied_vol import ImpliedVolSolver, IVStatus


class DataQuality(str, Enum):
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
        chain: "pd.DataFrame",
        spot: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        underlying: str = "UNKNOWN",
        observed_at: str = "",
        min_tte: float = 1/365,  # Minimum 1 day to expiry
        iv_bounds: tuple[float, float] = (0.01, 3.0),
    ) -> "VolatilitySurface":
        """
        Construct surface from an options chain DataFrame.
        
        Expected columns:
            strike, expiration (or time_to_expiry), option_type,
            bid, ask (or mid_price)
        """
        import pandas as pd
        
        solver = ImpliedVolSolver()
        points = []
        warnings = []
        
        # Ensure we have required columns
        required = ['strike', 'option_type']
        has_tte = 'time_to_expiry' in chain.columns
        has_exp = 'expiration' in chain.columns
        has_bid_ask = 'bid' in chain.columns and 'ask' in chain.columns
        has_mid = 'mid_price' in chain.columns
        
        if not all(col in chain.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")
        if not (has_tte or has_exp):
            raise ValueError("Need either 'time_to_expiry' or 'expiration' column")
        if not (has_bid_ask or has_mid):
            raise ValueError("Need either 'bid'/'ask' or 'mid_price' columns")
        
        for _, row in chain.iterrows():
            strike = float(row['strike'])
            
            # Get time to expiry
            if has_tte:
                tte = float(row['time_to_expiry'])
                expiry = tte  # Use TTE as expiry identifier
            else:
                # Calculate from expiration date
                exp_date = pd.to_datetime(row['expiration'])
                now = pd.Timestamp.now()
                tte = (exp_date - now).days / 365
                expiry = (exp_date - now).days
            
            if tte < min_tte:
                continue
            
            # Get option type
            opt_type_str = str(row['option_type']).lower()
            opt_type = OptionType.CALL if 'call' in opt_type_str else OptionType.PUT
            
            # Get price
            if has_bid_ask:
                bid = float(row.get('bid', 0) or 0)
                ask = float(row.get('ask', 0) or 0)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None
            else:
                mid = float(row['mid_price'])
                bid = ask = None
            
            if mid is None or mid <= 0:
                continue
            
            # Solve for IV
            result = solver.solve(
                mid, spot, strike, tte, risk_free_rate, opt_type, dividend_yield
            )
            
            if result.status != IVStatus.SUCCESS:
                continue
            
            iv = result.iv
            if iv is None or not (iv_bounds[0] <= iv <= iv_bounds[1]):
                continue
            
            # Calculate moneyness
            forward = spot * np.exp((risk_free_rate - dividend_yield) * tte)
            moneyness = np.log(strike / forward)
            
            # Bid/Ask IV if available
            bid_iv = ask_iv = None
            if bid is not None and bid > 0:
                bid_result = solver.solve(
                    bid, spot, strike, tte, risk_free_rate, opt_type, dividend_yield
                )
                if bid_result.status == IVStatus.SUCCESS:
                    bid_iv = bid_result.iv
            if ask is not None and ask > 0:
                ask_result = solver.solve(
                    ask, spot, strike, tte, risk_free_rate, opt_type, dividend_yield
                )
                if ask_result.status == IVStatus.SUCCESS:
                    ask_iv = ask_result.iv
            
            points.append(SurfacePoint(
                strike=strike,
                expiry=expiry,
                iv=iv,
                quality=DataQuality.OBSERVED,
                moneyness=moneyness,
                bid_iv=bid_iv,
                ask_iv=ask_iv,
            ))
        
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
                coords, ivs, kernel='thin_plate_spline', smoothing=0.001
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
        tte = expiry / 365 if expiry > 1 else expiry
        forward = spot * np.exp(
            (self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte
        )
        moneyness = np.log(strike / forward)
        
        # Check if in extrapolation region
        all_moneyness = [p.moneyness for p in self.points]
        all_expiry = [p.expiry for p in self.points]
        
        is_extrapolated = (
            moneyness < min(all_moneyness) - 0.05 or
            moneyness > max(all_moneyness) + 0.05 or
            expiry < min(all_expiry) * 0.9 or
            expiry > max(all_expiry) * 1.1
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
        tte = expiry / 365 if expiry > 1 else expiry
        forward = spot * np.exp(
            (self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte
        )
        
        # Generate moneyness grid
        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        strikes = forward * np.exp(moneyness)
        
        ivs = []
        qualities = []
        
        for k, m in zip(strikes, moneyness):
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
        expiries = sorted(set(p.expiry for p in self.points))
        
        term_structure = []
        spot = self.metadata.spot
        
        for expiry in expiries:
            tte = expiry / 365 if expiry > 1 else expiry
            forward = spot * np.exp(
                (self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte
            )
            strike = forward * np.exp(moneyness)
            
            iv, quality = self.get_iv(strike, expiry)
            if iv is not None:
                term_structure.append(TermStructurePoint(
                    expiry=expiry,
                    time_to_expiry=tte,
                    atm_iv=iv,
                    atm_strike=strike,
                    quality=quality,
                ))
        
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
        expiries = sorted(set(p.expiry for p in self.points))
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
                tte = expiry / 365 if expiry > 1 else expiry
                forward = spot * np.exp(
                    (self.metadata.risk_free_rate - self.metadata.dividend_yield) * tte
                )
                strike = forward * np.exp(moneyness)
                
                iv, quality = self.get_iv(strike, expiry)
                IV[i, j] = iv if iv is not None else np.nan
                
                quality_map = {
                    DataQuality.OBSERVED: 3,
                    DataQuality.INTERPOLATED: 2,
                    DataQuality.EXTRAPOLATED: 1,
                    DataQuality.SPARSE: 0,
                }
                quality_mask[i, j] = quality_map[quality]
        
        return {
            'moneyness': M,
            'expiry': E,
            'iv': IV,
            'quality_mask': quality_mask,
            'observed_points': [
                (p.moneyness, p.expiry, p.iv) for p in self.points
                if p.quality == DataQuality.OBSERVED
            ],
        }
    
    def to_dict(self) -> dict:
        """Convert surface to dictionary for JSON serialization."""
        return {
            'metadata': {
                'underlying': self.metadata.underlying,
                'spot': self.metadata.spot,
                'observed_at': self.metadata.observed_at,
                'risk_free_rate': self.metadata.risk_free_rate,
                'dividend_yield': self.metadata.dividend_yield,
                'n_observed_points': self.metadata.n_observed_points,
                'warnings': self.metadata.warnings,
            },
            'points': [
                {
                    'strike': p.strike,
                    'expiry': p.expiry,
                    'iv': p.iv,
                    'quality': p.quality.value,
                    'moneyness': p.moneyness,
                    'bid_iv': p.bid_iv,
                    'ask_iv': p.ask_iv,
                }
                for p in self.points
            ],
        }
