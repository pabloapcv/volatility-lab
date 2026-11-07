import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { volatility, data, SurfaceResponse, SmileResponse, TermStructureResponse } from '../api'
import { RefreshCw, AlertTriangle, CheckCircle2, Layers } from 'lucide-react'

type ViewMode = 'surface' | 'smile' | 'term'

export default function VolatilitySurfaceLab() {
  const [symbol, setSymbol] = useState('SPY')
  const [viewMode, setViewMode] = useState<ViewMode>('surface')
  const [selectedExpiry, setSelectedExpiry] = useState(30)

  const underlyingsQuery = useQuery({
    queryKey: ['underlyings'],
    queryFn: () => data.underlyings(),
  })

  const snapshotsQuery = useQuery({
    queryKey: ['snapshots', symbol],
    queryFn: () => data.snapshots(symbol),
  })

  const surfaceQuery = useQuery({
    queryKey: ['surface', symbol],
    queryFn: () => volatility.surface(symbol),
    enabled: snapshotsQuery.data && snapshotsQuery.data.data.total > 0,
  })

  const smileQuery = useQuery({
    queryKey: ['smile', symbol, selectedExpiry],
    queryFn: () => volatility.smile(symbol, selectedExpiry),
    enabled: viewMode === 'smile' && snapshotsQuery.data && snapshotsQuery.data.data.total > 0,
  })

  const termQuery = useQuery({
    queryKey: ['term', symbol],
    queryFn: () => volatility.termStructure(symbol),
    enabled: viewMode === 'term' && snapshotsQuery.data && snapshotsQuery.data.data.total > 0,
  })

  const loadChainMutation = useMutation({
    mutationFn: (sym: string) => data.chain(sym, true),
    onSuccess: () => {
      snapshotsQuery.refetch()
      surfaceQuery.refetch()
    },
  })

  const hasData = snapshotsQuery.data && snapshotsQuery.data.data.total > 0

  const renderSurface = (surfaceData: SurfaceResponse) => {
    if (!surfaceData.grid) {
      return (
        <div className="flex items-center justify-center h-96 text-gray-500">
          Not enough data points for surface interpolation
        </div>
      )
    }

    const { moneyness, expiry, iv, observed_points } = surfaceData.grid

    return (
      <Plot
        data={[
          {
            type: 'surface',
            x: moneyness[0],
            y: expiry.map(row => row[0]),
            z: iv,
            colorscale: 'Viridis',
            name: 'IV Surface',
            opacity: 0.9,
            showscale: true,
            colorbar: {
              title: { text: 'IV', side: 'right' },
            },
          },
          {
            type: 'scatter3d',
            mode: 'markers',
            x: observed_points.map(p => p[0]),
            y: observed_points.map(p => p[1]),
            z: observed_points.map(p => p[2]),
            marker: {
              size: 4,
              color: 'red',
              symbol: 'circle',
            },
            name: 'Observed Points',
          },
        ]}
        layout={{
          title: { text: `${symbol} Implied Volatility Surface` },
          autosize: true,
          height: 500,
          margin: { l: 0, r: 0, t: 40, b: 0 },
          scene: {
            xaxis: { title: { text: 'Log Moneyness' } },
            yaxis: { title: { text: 'Days to Expiry' } },
            zaxis: { title: { text: 'Implied Vol' } },
            camera: {
              eye: { x: 1.5, y: 1.5, z: 0.8 },
            },
          },
          legend: { x: 0, y: 1 },
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    )
  }

  const renderSmile = (smileData: SmileResponse) => {
    const validIndices = smileData.ivs.map((iv, i) => iv !== null ? i : -1).filter(i => i >= 0)
    
    return (
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: validIndices.map(i => smileData.moneyness[i]),
            y: validIndices.map(i => smileData.ivs[i]),
            name: 'IV Smile',
            line: { color: '#0ea5e9', width: 2 },
            marker: { size: 6 },
          },
        ]}
        layout={{
          title: { text: `${symbol} Volatility Smile (${smileData.expiry} days)` },
          autosize: true,
          height: 400,
          xaxis: { title: { text: 'Log Moneyness (ln(K/F))' } },
          yaxis: { title: { text: 'Implied Volatility' }, tickformat: '.1%' },
          margin: { l: 60, r: 40, t: 60, b: 60 },
          annotations: smileData.atm_iv ? [
            {
              x: 0,
              y: smileData.atm_iv,
              text: `ATM: ${(smileData.atm_iv * 100).toFixed(1)}%`,
              showarrow: true,
              arrowhead: 2,
            },
          ] : [],
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    )
  }

  const renderTermStructure = (termData: TermStructureResponse) => {
    return (
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: termData.points.map(p => p.time_to_expiry * 365),
            y: termData.points.map(p => p.atm_iv),
            name: 'ATM Term Structure',
            line: { color: '#8b5cf6', width: 2 },
            marker: { size: 8 },
          },
        ]}
        layout={{
          title: { text: `${symbol} ATM Volatility Term Structure` },
          autosize: true,
          height: 400,
          xaxis: { title: { text: 'Days to Expiry' } },
          yaxis: { title: { text: 'ATM Implied Volatility' }, tickformat: '.1%' },
          margin: { l: 60, r: 40, t: 60, b: 60 },
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <label className="label">Underlying</label>
              <select
                className="input w-40"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
              >
                {underlyingsQuery.data?.data.underlyings.map((u) => (
                  <option key={u.symbol} value={u.symbol}>
                    {u.symbol}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">View</label>
              <div className="flex gap-1">
                <button
                  onClick={() => setViewMode('surface')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'surface'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  3D Surface
                </button>
                <button
                  onClick={() => setViewMode('smile')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'smile'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Smile
                </button>
                <button
                  onClick={() => setViewMode('term')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'term'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Term Structure
                </button>
              </div>
            </div>

            {viewMode === 'smile' && (
              <div>
                <label className="label">Expiry (days)</label>
                <input
                  type="number"
                  className="input w-24"
                  value={selectedExpiry}
                  onChange={(e) => setSelectedExpiry(parseInt(e.target.value) || 30)}
                  min={1}
                  max={365}
                />
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            {hasData ? (
              <span className="flex items-center gap-1 text-sm text-green-600">
                <CheckCircle2 className="w-4 h-4" />
                {snapshotsQuery.data?.data.snapshots.find((s) => !s.provider?.includes('synthetic'))?.valid_contracts
                  || snapshotsQuery.data?.data.snapshots[0]?.valid_contracts
                  || 0} contracts
              </span>
            ) : (
              <span className="flex items-center gap-1 text-sm text-amber-600">
                <AlertTriangle className="w-4 h-4" />
                No data loaded
              </span>
            )}

            <button
              onClick={() => loadChainMutation.mutate(symbol)}
              disabled={loadChainMutation.isPending}
              className="btn-primary flex items-center gap-2"
            >
              {loadChainMutation.isPending ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  Load Chain
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Chart Area */}
      <div className="card">
        {!hasData ? (
          <div className="flex flex-col items-center justify-center h-96 text-gray-500">
            <Layers className="w-12 h-12 mb-4 text-gray-400" />
            <p className="text-lg font-medium">No volatility data available</p>
            <p className="text-sm mt-1">Click "Load Chain" to fetch options data for {symbol}</p>
          </div>
        ) : viewMode === 'surface' && surfaceQuery.isLoading ? (
          <div className="flex items-center justify-center h-96">
            <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : viewMode === 'surface' && surfaceQuery.data ? (
          renderSurface(surfaceQuery.data.data)
        ) : viewMode === 'smile' && smileQuery.isLoading ? (
          <div className="flex items-center justify-center h-96">
            <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : viewMode === 'smile' && smileQuery.data ? (
          renderSmile(smileQuery.data.data)
        ) : viewMode === 'term' && termQuery.isLoading ? (
          <div className="flex items-center justify-center h-96">
            <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : viewMode === 'term' && termQuery.data ? (
          renderTermStructure(termQuery.data.data)
        ) : (
          <div className="flex items-center justify-center h-96 text-gray-500">
            Error loading data
          </div>
        )}
      </div>

      {/* Surface Info */}
      {surfaceQuery.data && (
        <div className="grid grid-cols-3 gap-4">
          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 mb-1">Data Points</h3>
            <div className="text-2xl font-bold text-gray-900">
              {surfaceQuery.data.data.metadata.n_observed_points}
            </div>
            <div className="text-sm text-gray-500">observed IV points</div>
          </div>

          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 mb-1">Spot Price</h3>
            <div className="text-2xl font-bold text-gray-900">
              ${surfaceQuery.data.data.metadata.spot.toFixed(2)}
            </div>
            <div className="text-sm text-gray-500">{symbol}</div>
          </div>

          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 mb-1">Last Updated</h3>
            <div className="text-lg font-bold text-gray-900">
              {surfaceQuery.data.data.metadata.observed_at 
                ? new Date(surfaceQuery.data.data.metadata.observed_at).toLocaleString()
                : 'Unknown'}
            </div>
            <div className="text-sm text-gray-500">observation time</div>
          </div>
        </div>
      )}

      {surfaceQuery.data?.data.metadata && (
        <div className={`card ${surfaceQuery.data.data.metadata.is_synthetic ? 'bg-amber-50 border-amber-200' : 'bg-sky-50 border-sky-200'}`}>
          <h3 className="text-sm font-medium text-gray-800 mb-1">Data source</h3>
          <p className="text-sm text-gray-700">
            {surfaceQuery.data.data.metadata.source_label
              || (surfaceQuery.data.data.metadata.is_synthetic
                ? 'Synthetic demo fallback — not live market quotes'
                : `yfinance · ${String(surfaceQuery.data.data.metadata.observed_at || '').slice(0, 10)}`)}
          </p>
          {surfaceQuery.data.data.metadata.note && (
            <p className="text-xs text-gray-500 mt-1">{surfaceQuery.data.data.metadata.note}</p>
          )}
          {surfaceQuery.data.data.metadata.is_synthetic && (
            <p className="text-xs text-amber-700 mt-1">
              A real yfinance snapshot is preferred when present. This synthetic chain is the offline fallback.
            </p>
          )}
        </div>
      )}

      {/* Warnings */}
      {surfaceQuery.data?.data.warnings && surfaceQuery.data.data.warnings.length > 0 && (
        <div className="card bg-amber-50 border-amber-200">
          <h3 className="text-sm font-medium text-amber-800 mb-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Data Quality Warnings
          </h3>
          <ul className="text-sm text-amber-700 space-y-1">
            {surfaceQuery.data.data.warnings.map((warning, i) => (
              <li key={i}>• {warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Legend */}
      {viewMode === 'surface' && hasData && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Surface Legend</h3>
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500" />
              <span className="text-gray-600">Observed IV points (market quotes)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-gradient-to-r from-purple-500 to-yellow-400" />
              <span className="text-gray-600">Interpolated surface (RBF)</span>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            The 3D surface is interpolated using thin-plate spline RBF. Points outside the observed 
            strike/expiry range are extrapolated and should be interpreted with caution.
          </p>
        </div>
      )}
    </div>
  )
}
