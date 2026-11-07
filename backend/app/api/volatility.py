"""Volatility API endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.schemas.pricing import OptionType
from backend.app.schemas.volatility import (
    ArbitrageBounds,
    BidAskIVRequest,
    BidAskIVResponse,
    DataQuality,
    ImpliedVolRequest,
    ImpliedVolResponse,
    IVStatus,
    SmileResponse,
    SurfaceGridResponse,
    SurfacePoint,
    SurfaceRequest,
    SurfaceResponse,
    TermStructurePoint,
    TermStructureResponse,
)
from volatilitylab.data.snapshots import SnapshotManager
from volatilitylab.pricing.black_scholes import OptionType as BSOptionType
from volatilitylab.volatility.implied_vol import ImpliedVolSolver
from volatilitylab.volatility.implied_vol import IVStatus as InternalIVStatus
from volatilitylab.volatility.surface import DataQuality as InternalDataQuality
from volatilitylab.volatility.surface import VolatilitySurface

router = APIRouter(prefix="/volatility", tags=["volatility"])

SNAPSHOT_DIR = Path("./data/snapshots")


def _convert_option_type(opt_type: OptionType) -> BSOptionType:
    return BSOptionType.CALL if opt_type == OptionType.CALL else BSOptionType.PUT


def _convert_iv_status(status: InternalIVStatus) -> IVStatus:
    return IVStatus(status.value)


def _convert_data_quality(quality: InternalDataQuality) -> DataQuality:
    return DataQuality(quality.value)


@router.post("/implied", response_model=ImpliedVolResponse)
async def calculate_implied_volatility(request: ImpliedVolRequest) -> ImpliedVolResponse:
    """
    Calculate implied volatility from market price.
    
    Uses Brent's method for root finding. Validates against no-arbitrage
    bounds and returns status indicating success or failure reason.
    """
    try:
        solver = ImpliedVolSolver()
        result = solver.solve(
            price=request.price,
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            risk_free_rate=request.risk_free_rate,
            option_type=_convert_option_type(request.option_type),
            dividend_yield=request.dividend_yield,
        )
        
        return ImpliedVolResponse(
            implied_volatility=result.iv,
            status=_convert_iv_status(result.status),
            price_error=result.price_error,
            iterations=result.iterations,
            bounds=ArbitrageBounds(
                lower=result.bounds.lower,
                upper=result.bounds.upper,
                forward=result.bounds.forward,
                discount_factor=result.bounds.discount_factor,
            ),
            message=result.message,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@router.post("/implied/bid-ask", response_model=BidAskIVResponse)
async def calculate_bid_ask_iv(request: BidAskIVRequest) -> BidAskIVResponse:
    """
    Calculate implied volatility for bid, ask, and mid prices.
    
    Returns separate IV values and statuses for each price level.
    """
    try:
        solver = ImpliedVolSolver()
        results = solver.solve_bid_ask_mid(
            bid=request.bid,
            ask=request.ask,
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            risk_free_rate=request.risk_free_rate,
            option_type=_convert_option_type(request.option_type),
            dividend_yield=request.dividend_yield,
        )
        
        return BidAskIVResponse(
            bid_iv=results['bid_iv'].iv,
            ask_iv=results['ask_iv'].iv,
            mid_iv=results['mid_iv'].iv,
            bid_status=_convert_iv_status(results['bid_iv'].status),
            ask_status=_convert_iv_status(results['ask_iv'].status),
            mid_status=_convert_iv_status(results['mid_iv'].status),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@router.get("/surface/{symbol}", response_model=SurfaceResponse)
async def get_volatility_surface(
    symbol: str,
    snapshot_id: str | None = None,
    include_grid: bool = True,
    n_strikes: int = 20,
    n_expiries: int = 10,
) -> SurfaceResponse:
    """
    Get volatility surface for a symbol.
    
    Uses saved snapshot if snapshot_id provided, otherwise loads latest
    available snapshot. Returns observed points and optionally an
    interpolated grid for 3D visualization.
    """
    try:
        manager = SnapshotManager(SNAPSHOT_DIR)
        
        if snapshot_id:
            snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}_cleaned.parquet"
            if not snapshot_path.exists():
                raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
            from volatilitylab.data.snapshots import load_snapshot
            df, metadata = load_snapshot(snapshot_path)
        else:
            result = manager.load_latest(symbol.upper())
            if result is None:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No snapshots available for {symbol}. Load chain first."
                )
            df, metadata = result
        
        # Build surface
        surface = VolatilitySurface.from_option_chain(
            df,
            spot=metadata.get('spot', 100),
            risk_free_rate=0.045,
            underlying=symbol.upper(),
            observed_at=metadata.get('observed_at', ''),
        )
        
        # Convert points
        points = [
            SurfacePoint(
                strike=p.strike,
                expiry=p.expiry,
                iv=p.iv,
                quality=_convert_data_quality(p.quality),
                moneyness=p.moneyness,
                bid_iv=p.bid_iv,
                ask_iv=p.ask_iv,
            )
            for p in surface.points
        ]
        
        # Generate grid if requested
        grid = None
        if include_grid and len(surface.points) >= 4:
            grid_data = surface.get_surface_grid(
                n_strikes=n_strikes,
                n_expiries=n_expiries,
            )
            grid = SurfaceGridResponse(
                moneyness=grid_data['moneyness'].tolist(),
                expiry=grid_data['expiry'].tolist(),
                iv=[[float(v) if not (v != v) else None for v in row] for row in grid_data['iv']],
                quality_mask=grid_data['quality_mask'].tolist(),
                observed_points=grid_data['observed_points'],
            )
        
        return SurfaceResponse(
            metadata=surface.metadata.__dict__,
            points=points,
            grid=grid,
            warnings=surface.metadata.warnings,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building surface: {str(e)}")


@router.get("/smile/{symbol}/{expiry_days}", response_model=SmileResponse)
async def get_volatility_smile(
    symbol: str,
    expiry_days: int,
    snapshot_id: str | None = None,
    n_strikes: int = 20,
) -> SmileResponse:
    """
    Get volatility smile at a specific expiry.
    
    Returns strikes, IVs, moneyness, and quality flags for each point,
    plus ATM IV, skew, and convexity estimates.
    """
    try:
        manager = SnapshotManager(SNAPSHOT_DIR)
        
        if snapshot_id:
            from volatilitylab.data.snapshots import load_snapshot
            snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}_cleaned.parquet"
            df, metadata = load_snapshot(snapshot_path)
        else:
            result = manager.load_latest(symbol.upper())
            if result is None:
                raise HTTPException(status_code=404, detail=f"No snapshots for {symbol}")
            df, metadata = result
        
        surface = VolatilitySurface.from_option_chain(
            df, spot=metadata.get('spot', 100), underlying=symbol.upper()
        )
        
        smile = surface.get_smile(expiry=expiry_days, n_strikes=n_strikes)
        
        return SmileResponse(
            expiry=smile.expiry,
            time_to_expiry=smile.time_to_expiry,
            strikes=smile.strikes.tolist(),
            ivs=[float(v) if not (v != v) else None for v in smile.ivs],
            moneyness=smile.moneyness.tolist(),
            qualities=[_convert_data_quality(q) for q in smile.qualities],
            atm_iv=smile.atm_iv,
            skew=smile.skew,
            convexity=smile.convexity,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/term-structure/{symbol}", response_model=TermStructureResponse)
async def get_term_structure(
    symbol: str,
    snapshot_id: str | None = None,
    moneyness: float = 0.0,
) -> TermStructureResponse:
    """
    Get ATM (or fixed moneyness) volatility term structure.
    
    Returns IV at each available expiry sorted by time to expiration.
    """
    try:
        manager = SnapshotManager(SNAPSHOT_DIR)
        
        if snapshot_id:
            from volatilitylab.data.snapshots import load_snapshot
            snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}_cleaned.parquet"
            df, metadata = load_snapshot(snapshot_path)
        else:
            result = manager.load_latest(symbol.upper())
            if result is None:
                raise HTTPException(status_code=404, detail=f"No snapshots for {symbol}")
            df, metadata = result
        
        surface = VolatilitySurface.from_option_chain(
            df, spot=metadata.get('spot', 100), underlying=symbol.upper()
        )
        
        term_structure = surface.get_term_structure(moneyness=moneyness)
        
        return TermStructureResponse(
            points=[
                TermStructurePoint(
                    expiry=p.expiry,
                    time_to_expiry=p.time_to_expiry,
                    atm_iv=p.atm_iv,
                    atm_strike=p.atm_strike,
                    quality=_convert_data_quality(p.quality),
                )
                for p in term_structure
            ],
            underlying=symbol.upper(),
            spot=metadata.get('spot', 100),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
