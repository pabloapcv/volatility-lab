import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface BlackScholesRequest {
  spot: number
  strike: number
  time_to_expiry: number
  risk_free_rate: number
  volatility: number
  option_type: 'call' | 'put'
  dividend_yield?: number
}

export interface Greeks {
  delta: number
  gamma: number
  vega: number
  theta: number
  rho: number
}

export interface BlackScholesResponse {
  price: number
  greeks: Greeks
  d1: number
  d2: number
  model_id: string
  assumptions: string[]
  warnings?: string[]
}

export interface BinomialRequest extends BlackScholesRequest {
  steps?: number
  exercise_style?: 'european' | 'american'
}

export interface BinomialResponse {
  price: number
  greeks: Greeks
  steps: number
  exercise_style: string
  early_exercise_nodes: number
  convergence_estimate: number | null
  model_id: string
  warnings?: string[]
}

export interface MonteCarloRequest extends BlackScholesRequest {
  n_paths?: number
  n_steps?: number
  seed?: number
  variance_reduction?: 'none' | 'antithetic' | 'control_variate' | 'both'
  confidence_level?: number
}

export interface MonteCarloResponse {
  price: number
  std_error: number
  confidence_interval: {
    lower: number
    upper: number
    level: number
  }
  n_paths: number
  n_steps: number
  seed: number | null
  variance_reduction: string
  model_id: string
  warnings?: string[]
}

export interface ImpliedVolRequest {
  price: number
  spot: number
  strike: number
  time_to_expiry: number
  risk_free_rate: number
  option_type: 'call' | 'put'
  dividend_yield?: number
}

export interface ImpliedVolResponse {
  implied_volatility: number | null
  status: string
  price_error: number | null
  iterations: number
  bounds: {
    lower: number
    upper: number
    forward: number
    discount_factor: number
  }
  message?: string
}

export interface SurfacePoint {
  strike: number
  expiry: number
  iv: number
  quality: string
  moneyness: number
  bid_iv?: number
  ask_iv?: number
}

export interface SurfaceResponse {
  metadata: {
    underlying: string
    spot: number
    observed_at: string
    n_observed_points: number
    warnings: string[]
  }
  points: SurfacePoint[]
  grid?: {
    moneyness: number[][]
    expiry: number[][]
    iv: (number | null)[][]
    quality_mask: number[][]
    observed_points: [number, number, number][]
  }
  warnings: string[]
}

export interface SmileResponse {
  expiry: number
  time_to_expiry: number
  strikes: number[]
  ivs: (number | null)[]
  moneyness: number[]
  qualities: string[]
  atm_iv: number | null
  skew: number | null
  convexity: number | null
}

export interface TermStructureResponse {
  points: {
    expiry: number
    time_to_expiry: number
    atm_iv: number
    atm_strike: number
    quality: string
  }[]
  underlying: string
  spot: number
}

export interface SnapshotInfo {
  snapshot_id: string
  symbol: string
  provider: string
  observed_at: string
  spot: number
  total_contracts: number
  valid_contracts: number
}

// API functions
export const pricing = {
  blackScholes: (data: BlackScholesRequest) => 
    api.post<BlackScholesResponse>('/pricing/black-scholes', data),
  
  binomial: (data: BinomialRequest) => 
    api.post<BinomialResponse>('/pricing/binomial', data),
  
  monteCarlo: (data: MonteCarloRequest) => 
    api.post<MonteCarloResponse>('/pricing/monte-carlo', data),
  
  compare: (data: BlackScholesRequest) => 
    api.post<{
      black_scholes: { price: number; delta: number }
      binomial_100: { price: number; delta: number; error_vs_bs: number }
      binomial_500: { price: number; delta: number; error_vs_bs: number }
      monte_carlo: { price: number; std_error: number; ci_95: number[]; error_vs_bs: number }
      max_discrepancy: number
    }>('/pricing/compare', data),
}

export const volatility = {
  impliedVol: (data: ImpliedVolRequest) => 
    api.post<ImpliedVolResponse>('/volatility/implied', data),
  
  surface: (symbol: string, includeGrid = true) => 
    api.get<SurfaceResponse>(`/volatility/surface/${symbol}?include_grid=${includeGrid}`),
  
  smile: (symbol: string, expiryDays: number) => 
    api.get<SmileResponse>(`/volatility/smile/${symbol}/${expiryDays}`),
  
  termStructure: (symbol: string) => 
    api.get<TermStructureResponse>(`/volatility/term-structure/${symbol}`),
}

export const data = {
  underlyings: () => 
    api.get<{ underlyings: { symbol: string; name: string; type: string }[] }>('/data/underlyings'),
  
  chain: (symbol: string, saveSnapshot = false) => 
    api.post('/data/chain', { symbol, save_snapshot: saveSnapshot }),
  
  snapshots: (symbol?: string) => 
    api.get<{ snapshots: SnapshotInfo[]; total: number }>(
      `/data/snapshots${symbol ? `?symbol=${symbol}` : ''}`
    ),
}
