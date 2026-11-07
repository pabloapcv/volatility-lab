#!/usr/bin/env python3
"""
Create a sample options chain snapshot for offline testing.

This script generates synthetic option chain data that mimics real market data
structure, allowing tests and demos to run without network access.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_chain(
    symbol: str = "SPY",
    spot: float = 585.0,
    n_expirations: int = 6,
    strikes_per_expiration: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic options chain data."""
    
    np.random.seed(42)  # Reproducibility
    
    observed_at = datetime.utcnow()
    
    # Generate expirations (weekly and monthly)
    expirations = []
    base_date = datetime.now()
    for i in range(n_expirations):
        exp = base_date + timedelta(days=7 * (i + 1))
        expirations.append(exp.strftime("%Y-%m-%d"))
    
    rows = []
    
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        tte = (exp_date - datetime.now()).days / 365
        tte = max(tte, 0.01)
        
        # ATM vol around 15-20% for SPY
        atm_vol = 0.16 + np.random.uniform(-0.02, 0.02)
        
        # Generate strikes around spot
        strike_range = spot * 0.15  # ±15%
        strikes = np.linspace(spot - strike_range, spot + strike_range, strikes_per_expiration)
        
        for strike in strikes:
            for opt_type in ['call', 'put']:
                # Moneyness-dependent vol (smile)
                moneyness = np.log(strike / spot)
                vol = atm_vol + 0.05 * moneyness**2 - 0.02 * moneyness  # Skew + smile
                vol = max(0.08, min(0.50, vol))
                
                # Simple BS-like price approximation
                from scipy.stats import norm
                
                d1 = (np.log(spot / strike) + (0.045 + vol**2/2) * tte) / (vol * np.sqrt(tte))
                d2 = d1 - vol * np.sqrt(tte)
                
                if opt_type == 'call':
                    price = spot * norm.cdf(d1) - strike * np.exp(-0.045 * tte) * norm.cdf(d2)
                else:
                    price = strike * np.exp(-0.045 * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
                
                price = max(0.01, price)
                
                # Add bid-ask spread
                spread = price * np.random.uniform(0.02, 0.08)
                bid = max(0.01, price - spread/2)
                ask = price + spread/2
                
                # Volume and OI based on moneyness
                atm_factor = np.exp(-10 * moneyness**2)
                volume = int(100 + 5000 * atm_factor * np.random.random())
                oi = int(500 + 20000 * atm_factor * np.random.random())
                
                rows.append({
                    'contract_symbol': f"{symbol}{exp.replace('-', '')}{opt_type[0].upper()}{int(strike*1000):08d}",
                    'strike': strike,
                    'expiration': exp_date,
                    'option_type': opt_type,
                    'bid': round(bid, 2),
                    'ask': round(ask, 2),
                    'last_price': round(price, 2),
                    'mid_price': round((bid + ask) / 2, 2),
                    'volume': volume,
                    'open_interest': oi,
                    'time_to_expiry': tte,
                    'moneyness': spot / strike,
                    'log_moneyness': moneyness,
                    'yf_implied_vol': vol,
                    'observed_at': observed_at,
                    'quality': 'valid' if bid > 0.05 and oi > 10 else 'low_open_interest',
                    'quality_reason': '',
                })
    
    df = pd.DataFrame(rows)
    
    metadata = {
        'symbol': symbol,
        'provider': 'synthetic',
        'spot': spot,
        'observed_at': observed_at.isoformat(),
        'expirations': expirations,
        'total_contracts': len(df),
        'valid_contracts': len(df[df['quality'] == 'valid']),
        'version': '1.0',
        'note': 'Synthetic data for testing - not real market quotes',
    }
    
    return df, metadata


def main():
    """Create sample snapshots."""
    snapshot_dir = Path(__file__).parent.parent / "data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SPY sample
    df, metadata = generate_sample_chain("SPY", spot=585.0)
    snapshot_id = "SPY_sample_offline"
    
    df.to_parquet(snapshot_dir / f"{snapshot_id}_cleaned.parquet", index=False)
    
    with open(snapshot_dir / f"{snapshot_id}_metadata.json", 'w') as f:
        json.dump(metadata | {'snapshot_id': snapshot_id, 'ingested_at': datetime.utcnow().isoformat()}, f, indent=2, default=str)
    
    print(f"Created sample snapshot: {snapshot_id}")
    print(f"  Contracts: {len(df)}")
    print(f"  Valid: {len(df[df['quality'] == 'valid'])}")
    print(f"  Location: {snapshot_dir / snapshot_id}_cleaned.parquet")
    
    # Generate QQQ sample
    df_qqq, meta_qqq = generate_sample_chain("QQQ", spot=495.0)
    snapshot_id_qqq = "QQQ_sample_offline"
    
    df_qqq.to_parquet(snapshot_dir / f"{snapshot_id_qqq}_cleaned.parquet", index=False)
    
    with open(snapshot_dir / f"{snapshot_id_qqq}_metadata.json", 'w') as f:
        json.dump(meta_qqq | {'snapshot_id': snapshot_id_qqq, 'ingested_at': datetime.utcnow().isoformat()}, f, indent=2, default=str)
    
    print(f"Created sample snapshot: {snapshot_id_qqq}")


if __name__ == "__main__":
    main()
