"""Pricing API endpoints."""

from fastapi import APIRouter, HTTPException

from backend.app.schemas.pricing import (
    BlackScholesRequest,
    BlackScholesResponse,
    BinomialRequest,
    BinomialResponse,
    ConfidenceInterval,
    GreeksResponse,
    MonteCarloRequest,
    MonteCarloResponse,
)
from volatilitylab.pricing.black_scholes import (
    BlackScholesModel,
    OptionType as BSOptionType,
)
from volatilitylab.pricing.binomial import (
    BinomialModel,
    ExerciseStyle as BinomialExerciseStyle,
)
from volatilitylab.pricing.monte_carlo import (
    MonteCarloModel,
    VarianceReduction as MCVarianceReduction,
)

router = APIRouter(prefix="/pricing", tags=["pricing"])


def _convert_option_type(opt_type: str) -> BSOptionType:
    """Convert API option type to internal type."""
    return BSOptionType.CALL if opt_type == "call" else BSOptionType.PUT


@router.post("/black-scholes", response_model=BlackScholesResponse)
async def calculate_black_scholes(request: BlackScholesRequest) -> BlackScholesResponse:
    """
    Calculate option price and Greeks using Black-Scholes-Merton model.
    
    Returns theoretical price, analytical Greeks (Delta, Gamma, Vega, Theta, Rho),
    d1/d2 parameters, model assumptions, and any numerical warnings.
    """
    try:
        model = BlackScholesModel()
        result = model.price(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            risk_free_rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=_convert_option_type(request.option_type.value),
            dividend_yield=request.dividend_yield,
        )
        
        return BlackScholesResponse(
            price=result.price,
            greeks=GreeksResponse(
                delta=result.greeks.delta,
                gamma=result.greeks.gamma,
                vega=result.greeks.vega,
                theta=result.greeks.theta,
                rho=result.greeks.rho,
            ),
            d1=result.d1,
            d2=result.d2,
            warnings=result.warnings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@router.post("/binomial", response_model=BinomialResponse)
async def calculate_binomial(request: BinomialRequest) -> BinomialResponse:
    """
    Calculate option price using CRR binomial tree model.
    
    Supports European and American exercise. Returns price, numerical Greeks,
    convergence estimate, and early exercise information for American options.
    """
    try:
        model = BinomialModel(steps=request.steps)
        
        exercise = (
            BinomialExerciseStyle.AMERICAN 
            if request.exercise_style.value == "american" 
            else BinomialExerciseStyle.EUROPEAN
        )
        
        result = model.price(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            risk_free_rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=_convert_option_type(request.option_type.value),
            dividend_yield=request.dividend_yield,
            exercise_style=exercise,
        )
        
        return BinomialResponse(
            price=result.price,
            greeks=GreeksResponse(
                delta=result.greeks.delta,
                gamma=result.greeks.gamma,
                vega=result.greeks.vega,
                theta=result.greeks.theta,
                rho=result.greeks.rho,
            ),
            steps=result.steps,
            exercise_style=request.exercise_style,
            early_exercise_nodes=result.early_exercise_nodes,
            convergence_estimate=result.convergence_estimate,
            warnings=result.warnings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def calculate_monte_carlo(request: MonteCarloRequest) -> MonteCarloResponse:
    """
    Calculate option price using Monte Carlo simulation.
    
    Supports antithetic and control variates for variance reduction.
    Returns price estimate, standard error, confidence interval, and
    convergence diagnostics.
    """
    try:
        var_red_map = {
            "none": MCVarianceReduction.NONE,
            "antithetic": MCVarianceReduction.ANTITHETIC,
            "control_variate": MCVarianceReduction.CONTROL_VARIATE,
            "both": MCVarianceReduction.BOTH,
        }
        
        model = MonteCarloModel(
            n_paths=request.n_paths,
            n_steps=request.n_steps,
            seed=request.seed,
            variance_reduction=var_red_map[request.variance_reduction.value],
        )
        
        result = model.price(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            risk_free_rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=_convert_option_type(request.option_type.value),
            dividend_yield=request.dividend_yield,
            confidence_level=request.confidence_level,
        )
        
        return MonteCarloResponse(
            price=result.price,
            std_error=result.std_error,
            confidence_interval=ConfidenceInterval(
                lower=result.confidence_interval.lower,
                upper=result.confidence_interval.upper,
                level=result.confidence_interval.level,
            ),
            n_paths=result.n_paths,
            n_steps=result.n_steps,
            seed=result.seed,
            variance_reduction=request.variance_reduction,
            control_variate_beta=result.control_variate_beta,
            warnings=result.warnings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@router.post("/compare")
async def compare_models(request: BlackScholesRequest) -> dict:
    """
    Compare all three pricing models for the same option.
    
    Returns Black-Scholes, Binomial (100 and 500 steps), and Monte Carlo
    prices with discrepancies highlighted.
    """
    try:
        bs_model = BlackScholesModel()
        bin_model = BinomialModel()
        mc_model = MonteCarloModel(seed=42)
        
        opt_type = _convert_option_type(request.option_type.value)
        
        bs_result = bs_model.price(
            request.spot, request.strike, request.time_to_expiry,
            request.risk_free_rate, request.volatility, opt_type, request.dividend_yield
        )
        
        bin_100 = bin_model.price(
            request.spot, request.strike, request.time_to_expiry,
            request.risk_free_rate, request.volatility, opt_type, request.dividend_yield,
            steps=100
        )
        
        bin_500 = bin_model.price(
            request.spot, request.strike, request.time_to_expiry,
            request.risk_free_rate, request.volatility, opt_type, request.dividend_yield,
            steps=500
        )
        
        mc_result = mc_model.price(
            request.spot, request.strike, request.time_to_expiry,
            request.risk_free_rate, request.volatility, opt_type, request.dividend_yield,
            n_paths=100000
        )
        
        return {
            "black_scholes": {
                "price": bs_result.price,
                "delta": bs_result.greeks.delta,
            },
            "binomial_100": {
                "price": bin_100.price,
                "delta": bin_100.greeks.delta,
                "error_vs_bs": abs(bin_100.price - bs_result.price),
            },
            "binomial_500": {
                "price": bin_500.price,
                "delta": bin_500.greeks.delta,
                "error_vs_bs": abs(bin_500.price - bs_result.price),
            },
            "monte_carlo": {
                "price": mc_result.price,
                "std_error": mc_result.std_error,
                "ci_95": [mc_result.confidence_interval.lower, mc_result.confidence_interval.upper],
                "error_vs_bs": abs(mc_result.price - bs_result.price),
            },
            "reference": "black_scholes",
            "max_discrepancy": max(
                abs(bin_100.price - bs_result.price),
                abs(bin_500.price - bs_result.price),
                abs(mc_result.price - bs_result.price),
            ),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")
