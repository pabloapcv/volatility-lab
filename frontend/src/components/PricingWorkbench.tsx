import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { pricing, BlackScholesRequest, BlackScholesResponse, BinomialResponse, MonteCarloResponse } from '../api'
import { Calculator, AlertCircle, CheckCircle, TrendingUp, TrendingDown } from 'lucide-react'
import UsageGuide from './UsageGuide'

interface PricingInputs {
  spot: number
  strike: number
  time_to_expiry: number
  risk_free_rate: number
  volatility: number
  option_type: 'call' | 'put'
  dividend_yield: number
  steps: number
  n_paths: number
}

interface PricingResults {
  bs?: BlackScholesResponse
  binomial?: BinomialResponse
  mc?: MonteCarloResponse
}

export default function PricingWorkbench() {
  const [inputs, setInputs] = useState<PricingInputs>({
    spot: 500,
    strike: 510,
    time_to_expiry: 0.25,
    risk_free_rate: 0.045,
    volatility: 0.22,
    option_type: 'call',
    dividend_yield: 0.01,
    steps: 200,
    n_paths: 100000,
  })

  const [results, setResults] = useState<PricingResults>({})

  const bsMutation = useMutation({
    mutationFn: (data: BlackScholesRequest) => pricing.blackScholes(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, bs: response.data }))
    },
  })

  const binomialMutation = useMutation({
    mutationFn: (data: BlackScholesRequest & { steps: number }) => pricing.binomial(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, binomial: response.data }))
    },
  })

  const mcMutation = useMutation({
    mutationFn: (data: BlackScholesRequest & { n_paths: number; seed: number }) => pricing.monteCarlo(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, mc: response.data }))
    },
  })

  const handleCalculate = () => {
    const baseRequest: BlackScholesRequest = {
      spot: inputs.spot,
      strike: inputs.strike,
      time_to_expiry: inputs.time_to_expiry,
      risk_free_rate: inputs.risk_free_rate,
      volatility: inputs.volatility,
      option_type: inputs.option_type,
      dividend_yield: inputs.dividend_yield,
    }

    bsMutation.mutate(baseRequest)
    binomialMutation.mutate({ ...baseRequest, steps: inputs.steps })
    mcMutation.mutate({ ...baseRequest, n_paths: inputs.n_paths, seed: 42 })
  }

  const isLoading = bsMutation.isPending || binomialMutation.isPending || mcMutation.isPending

  const formatPrice = (price: number | undefined) => 
    price !== undefined ? `$${price.toFixed(4)}` : '-'

  const formatGreek = (value: number | undefined) => 
    value !== undefined ? value.toFixed(4) : '-'

  return (
    <div className="space-y-6">
      <UsageGuide
        storageKey="pricing"
        items={[
          'Spot and strike are in dollars. Time to expiry is in years (0.25 ≈ three months). Volatility is annualized σ (0.22 = 22%). Risk-free rate and dividend yield are continuous decimals.',
          'Call vs Put sets the payoff. Click Calculate All Models to price with Black-Scholes, CRR Binomial, and Monte Carlo together.',
          'Black-Scholes is the closed-form European price. CRR Binomial is a recombining tree (European on this page). Monte Carlo simulates GBM paths (fixed seed 42).',
          'Binomial Steps and MC Paths only affect those two models — Black-Scholes ignores them.',
          'Greeks: Delta and Gamma are per $1 move in the underlying. Theta is daily time decay. Vega and Rho are per 1% change in volatility and in the interest rate.',
          'Try raising Binomial Steps (for example 100 → 500) and watch the binomial error vs Black-Scholes shrink.',
          'Monte Carlo shows a 95% confidence interval under the price — Black-Scholes should usually sit inside it.',
        ]}
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Input Panel */}
      <div className="card">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Calculator className="w-5 h-5 text-primary-600" />
          Option Parameters
        </h2>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Spot Price ($)</label>
              <input
                type="number"
                className="input"
                value={inputs.spot}
                onChange={(e) => setInputs({ ...inputs, spot: parseFloat(e.target.value) || 0 })}
              />
            </div>
            <div>
              <label className="label">Strike Price ($)</label>
              <input
                type="number"
                className="input"
                value={inputs.strike}
                onChange={(e) => setInputs({ ...inputs, strike: parseFloat(e.target.value) || 0 })}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Time to Expiry (years)</label>
              <input
                type="number"
                step="0.01"
                className="input"
                value={inputs.time_to_expiry}
                onChange={(e) => setInputs({ ...inputs, time_to_expiry: parseFloat(e.target.value) || 0 })}
              />
            </div>
            <div>
              <label className="label">Volatility (σ)</label>
              <input
                type="number"
                step="0.01"
                className="input"
                value={inputs.volatility}
                onChange={(e) => setInputs({ ...inputs, volatility: parseFloat(e.target.value) || 0 })}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Risk-Free Rate</label>
              <input
                type="number"
                step="0.001"
                className="input"
                value={inputs.risk_free_rate}
                onChange={(e) => setInputs({ ...inputs, risk_free_rate: parseFloat(e.target.value) || 0 })}
              />
            </div>
            <div>
              <label className="label">Dividend Yield</label>
              <input
                type="number"
                step="0.001"
                className="input"
                value={inputs.dividend_yield}
                onChange={(e) => setInputs({ ...inputs, dividend_yield: parseFloat(e.target.value) || 0 })}
              />
            </div>
          </div>

          <div>
            <label className="label">Option Type</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="optionType"
                  checked={inputs.option_type === 'call'}
                  onChange={() => setInputs({ ...inputs, option_type: 'call' })}
                  className="text-primary-600"
                />
                <span className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4 text-green-600" />
                  Call
                </span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="optionType"
                  checked={inputs.option_type === 'put'}
                  onChange={() => setInputs({ ...inputs, option_type: 'put' })}
                  className="text-primary-600"
                />
                <span className="flex items-center gap-1">
                  <TrendingDown className="w-4 h-4 text-red-600" />
                  Put
                </span>
              </label>
            </div>
          </div>

          <hr className="my-4" />

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Binomial Steps</label>
              <input
                type="number"
                className="input"
                value={inputs.steps}
                onChange={(e) => setInputs({ ...inputs, steps: parseInt(e.target.value) || 100 })}
              />
            </div>
            <div>
              <label className="label">MC Paths</label>
              <input
                type="number"
                className="input"
                value={inputs.n_paths}
                onChange={(e) => setInputs({ ...inputs, n_paths: parseInt(e.target.value) || 10000 })}
              />
            </div>
          </div>

          <button
            onClick={handleCalculate}
            disabled={isLoading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Calculating...
              </>
            ) : (
              <>
                <Calculator className="w-4 h-4" />
                Calculate All Models
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-2 space-y-6">
        {/* Price Comparison */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Model Comparison</h2>
          
          <div className="grid grid-cols-3 gap-4">
            {/* Black-Scholes */}
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-blue-900">Black-Scholes</h3>
                {results.bs && <CheckCircle className="w-4 h-4 text-green-600" />}
              </div>
              <div className="text-2xl font-bold text-blue-700">
                {formatPrice(results.bs?.price)}
              </div>
              <div className="text-sm text-blue-600 mt-1">Analytical solution</div>
              {results.bs?.warnings && (
                <div className="flex items-center gap-1 text-xs text-amber-600 mt-2">
                  <AlertCircle className="w-3 h-3" />
                  {results.bs.warnings[0]}
                </div>
              )}
            </div>

            {/* Binomial */}
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-purple-900">CRR Binomial</h3>
                {results.binomial && <CheckCircle className="w-4 h-4 text-green-600" />}
              </div>
              <div className="text-2xl font-bold text-purple-700">
                {formatPrice(results.binomial?.price)}
              </div>
              <div className="text-sm text-purple-600 mt-1">
                {results.binomial ? `${results.binomial.steps} steps` : 'Tree-based'}
              </div>
              {results.binomial?.convergence_estimate && (
                <div className="text-xs text-purple-500 mt-1">
                  Est. error: ${results.binomial.convergence_estimate.toFixed(6)}
                </div>
              )}
            </div>

            {/* Monte Carlo */}
            <div className="bg-emerald-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-emerald-900">Monte Carlo</h3>
                {results.mc && <CheckCircle className="w-4 h-4 text-green-600" />}
              </div>
              <div className="text-2xl font-bold text-emerald-700">
                {formatPrice(results.mc?.price)}
              </div>
              <div className="text-sm text-emerald-600 mt-1">
                {results.mc ? `±$${results.mc.std_error.toFixed(4)} (95% CI)` : 'Simulation'}
              </div>
              {results.mc && (
                <div className="text-xs text-emerald-500 mt-1">
                  [{results.mc.confidence_interval.lower.toFixed(4)}, {results.mc.confidence_interval.upper.toFixed(4)}]
                </div>
              )}
            </div>
          </div>

          {/* Discrepancy Analysis */}
          {results.bs && results.binomial && results.mc && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Model Discrepancies vs Black-Scholes</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Binomial Error:</span>
                  <span className="ml-2 font-mono">
                    ${Math.abs(results.binomial.price - results.bs.price).toFixed(6)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Monte Carlo Error:</span>
                  <span className="ml-2 font-mono">
                    ${Math.abs(results.mc.price - results.bs.price).toFixed(6)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Greeks Table */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Option Greeks</h2>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Greek</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Black-Scholes</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Binomial</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Interpretation</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100">
                  <td className="py-2 px-3 font-medium">Delta (Δ)</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.bs?.greeks.delta)}</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.binomial?.greeks.delta)}</td>
                  <td className="py-2 px-3 text-gray-500 text-xs">Change in option value per $1 change in underlying</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 px-3 font-medium">Gamma (Γ)</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.bs?.greeks.gamma)}</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.binomial?.greeks.gamma)}</td>
                  <td className="py-2 px-3 text-gray-500 text-xs">Rate of change of delta per $1 change in underlying</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 px-3 font-medium">Vega (ν)</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.bs?.greeks.vega)}</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.binomial?.greeks.vega)}</td>
                  <td className="py-2 px-3 text-gray-500 text-xs">Change in value per 1% change in volatility</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 px-3 font-medium">Theta (Θ)</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.bs?.greeks.theta)}</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.binomial?.greeks.theta)}</td>
                  <td className="py-2 px-3 text-gray-500 text-xs">Daily time decay</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-medium">Rho (ρ)</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.bs?.greeks.rho)}</td>
                  <td className="text-right py-2 px-3 font-mono">{formatGreek(results.binomial?.greeks.rho)}</td>
                  <td className="py-2 px-3 text-gray-500 text-xs">Change in value per 1% change in interest rate</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Model Assumptions */}
        {results.bs && (
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Model Assumptions</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Black-Scholes-Merton</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  {results.bs.assumptions.map((assumption, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-primary-600">•</span>
                      {assumption}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Numerical Parameters</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>d₁ = {results.bs.d1.toFixed(4)}</li>
                  <li>d₂ = {results.bs.d2.toFixed(4)}</li>
                  <li>N(d₁) = {(results.bs.greeks.delta / (inputs.option_type === 'call' ? 1 : -1)).toFixed(4)}</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}
