# Mathematical Models

This document provides the mathematical foundations for the pricing models implemented in VolatilityLab, including derivations, assumptions, and numerical tolerances.

## 1. Black-Scholes-Merton Model

### 1.1 Model Assumptions

The Black-Scholes-Merton model assumes:

1. **Log-normal price distribution**: The underlying asset follows geometric Brownian motion (GBM):
   $$dS_t = (r - q)S_t dt + \sigma S_t dW_t$$

2. **Constant volatility**: Volatility σ is constant over the option's life

3. **Continuous trading**: Markets allow continuous trading with no transaction costs

4. **No arbitrage**: Risk-free arbitrage opportunities are immediately eliminated

5. **European exercise**: Options can only be exercised at expiration

### 1.2 Pricing Formulas

For a European call option:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

For a European put option:

$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

Where:

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

And:
- $S$ = spot price
- $K$ = strike price
- $T$ = time to expiration (years)
- $r$ = continuously compounded risk-free rate
- $q$ = continuous dividend yield
- $σ$ = annualized volatility
- $N(·)$ = standard normal CDF

### 1.3 Analytical Greeks

**Delta (Δ)** - Sensitivity to underlying price:
- Call: $\Delta = e^{-qT} N(d_1)$
- Put: $\Delta = -e^{-qT} N(-d_1)$

**Gamma (Γ)** - Rate of change of delta:
$$\Gamma = \frac{e^{-qT} n(d_1)}{S \sigma \sqrt{T}}$$

**Vega (ν)** - Sensitivity to volatility (per 1% change):
$$\nu = S e^{-qT} \sqrt{T} \cdot n(d_1) \cdot 0.01$$

**Theta (Θ)** - Time decay (per calendar day):
$$\Theta_{call} = -\frac{S e^{-qT} n(d_1) \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$$

**Rho (ρ)** - Sensitivity to interest rate (per 1% change):
- Call: $\rho = KTe^{-rT}N(d_2) \cdot 0.01$
- Put: $\rho = -KTe^{-rT}N(-d_2) \cdot 0.01$

### 1.4 Numerical Tolerances

| Condition | Handling |
|-----------|----------|
| $T < 10^{-10}$ | Return intrinsic value |
| $\sigma < 10^{-10}$ | Return discounted intrinsic based on moneyness |
| $|d_1| > 8$ | Deep ITM/OTM warning |

### 1.5 Put-Call Parity

For European options, put-call parity must hold:

$$C - P = Se^{-qT} - Ke^{-rT}$$

This is verified in our test suite with tolerance $< 10^{-6}$.

---

## 2. Cox-Ross-Rubinstein Binomial Model

### 2.1 Model Construction

The CRR model discretizes the continuous GBM into a recombining binomial tree.

**Tree Parameters:**
- Time step: $\Delta t = T/N$ where $N$ is number of steps
- Up factor: $u = e^{\sigma\sqrt{\Delta t}}$
- Down factor: $d = 1/u = e^{-\sigma\sqrt{\Delta t}}$
- Risk-neutral probability: $p = \frac{e^{(r-q)\Delta t} - d}{u - d}$

### 2.2 Tree Construction

At time step $i$, node $j$ has asset price:

$$S_{i,j} = S_0 \cdot u^j \cdot d^{i-j}$$

where $j = 0, 1, ..., i$ represents the number of up moves.

### 2.3 Backward Induction

**European options:**

$$V_{i,j} = e^{-r\Delta t} [p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}]$$

**American options:**

$$V_{i,j} = \max\left(e^{-r\Delta t} [p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}], \text{Intrinsic}_{i,j}\right)$$

### 2.4 Numerical Greeks

Greeks are estimated from the tree structure using finite differences:

- **Delta**: From values at step 1: $\Delta = \frac{V_u - V_d}{S_u - S_d}$
- **Gamma**: From values at step 2: $\Gamma = \frac{\Delta_{up} - \Delta_{down}}{0.5(S_{uu} - S_{dd})}$
- **Vega, Theta, Rho**: Central differences by bumping parameters

### 2.5 Convergence

The binomial model converges to Black-Scholes as $N \to \infty$ with rate $O(1/N)$.

| Steps | Expected Error vs BS |
|-------|---------------------|
| 100 | ~$0.02 |
| 200 | ~$0.01 |
| 500 | ~$0.004 |

---

## 3. Monte Carlo Simulation

### 3.1 GBM Discretization

The exact discretization for GBM with constant parameters:

$$S_{t+\Delta t} = S_t \exp\left[(r - q - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right]$$

where $Z \sim N(0, 1)$.

### 3.2 European Option Pricing

The discounted expected payoff:

$$V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}\left[\max(S_T - K, 0)\right]$$

Estimated by:

$$\hat{V} = e^{-rT} \frac{1}{M} \sum_{i=1}^{M} \max(S_T^{(i)} - K, 0)$$

### 3.3 Standard Error

$$\text{SE} = \frac{\hat{\sigma}}{\sqrt{M}}$$

where $\hat{\sigma}$ is the sample standard deviation of discounted payoffs.

### 3.4 Confidence Intervals

95% confidence interval:

$$\left[\hat{V} - 1.96 \cdot \text{SE}, \hat{V} + 1.96 \cdot \text{SE}\right]$$

### 3.5 Variance Reduction

**Antithetic Variates:**

For each standard normal $Z_i$, also use $-Z_i$:

$$\hat{V}_{anti} = \frac{1}{2}(\hat{V}(Z) + \hat{V}(-Z))$$

Variance reduction factor typically 30-50% for option pricing.

**Control Variates:**

Using the underlying asset as control (known expectation):

$$\hat{V}_{cv} = \hat{V} - \beta(S_T - S_0 e^{(r-q)T})$$

where $\beta$ minimizes variance.

### 3.6 Convergence Diagnostics

- Track cumulative mean across batches
- Verify CI coverage through repeated experiments
- Effective sample size accounts for antithetic correlation

---

## 4. Implied Volatility

### 4.1 Definition

Implied volatility $\sigma_{IV}$ satisfies:

$$C_{market} = BS(S, K, T, r, \sigma_{IV}, q)$$

### 4.2 Solver Algorithm

We use Brent's method with:
- Search interval: $[10^{-6}, 5.0]$ (0.0001% to 500% annualized)
- Tolerance: $10^{-8}$ on price
- Maximum iterations: 100

### 4.3 No-Arbitrage Bounds

**Call bounds:**
$$\max(0, Se^{-qT} - Ke^{-rT}) \leq C \leq Se^{-qT}$$

**Put bounds:**
$$\max(0, Ke^{-rT} - Se^{-qT}) \leq P \leq Ke^{-rT}$$

Prices outside these bounds are rejected with `ARBITRAGE_VIOLATION`.

### 4.4 Round-Trip Accuracy

For any valid IV σ:

$$\left|\sigma - \text{IV}(\text{BS}(S, K, T, r, \sigma))\right| < 10^{-6}$$

---

## 5. Volatility Surface

### 5.1 Moneyness Definition

Log-moneyness relative to forward:

$$m = \ln(K/F)$$

where $F = Se^{(r-q)T}$ is the forward price.

### 5.2 Surface Interpolation

We use Radial Basis Function (RBF) interpolation with thin-plate spline kernel for smooth surfaces.

### 5.3 Smile Characteristics

**Skew**: Slope of IV vs moneyness at ATM
$$\text{Skew} = \frac{\partial \sigma_{IV}}{\partial m}\bigg|_{m=0}$$

**Convexity (Smile)**: Curvature at ATM
$$\text{Convexity} = \frac{\partial^2 \sigma_{IV}}{\partial m^2}\bigg|_{m=0}$$

### 5.4 Data Quality

Points are classified as:
- **Observed**: Direct market quote with valid IV solution
- **Interpolated**: Within observed strike/expiry range
- **Extrapolated**: Outside observed range (use with caution)
- **Sparse**: Insufficient nearby data for reliable interpolation

---

## 6. Numerical Stability

### 6.1 Edge Case Handling

| Condition | Risk | Mitigation |
|-----------|------|------------|
| $T \to 0$ | Division by zero | Return intrinsic value |
| $\sigma \to 0$ | Undefined $d_1$, $d_2$ | Use limiting behavior based on moneyness |
| $|d| \to \infty$ | $N(d) \approx 0$ or $1$ | Warning for precision loss |
| Deep OTM | Near-zero price | May lose precision in IV solver |

### 6.2 Test Coverage Targets

| Module | Target Coverage |
|--------|----------------|
| Black-Scholes | ≥ 90% |
| Binomial | ≥ 85% |
| Monte Carlo | ≥ 85% |
| Implied Vol | ≥ 90% |

---

## References

1. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.

2. Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics*, 7(3), 229-263.

3. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

4. Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
