import { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import UsageGuide from './UsageGuide';

interface MetricsRow {
  model: string;
  horizon: string;
  mae: number;
  rmse: number;
  qlike: number;
}

interface HypothesisTest {
  hypothesis: string;
  result: string;
}

interface DMComparison {
  model: string;
  dm_statistic: number;
  p_value: number;
  model_better: boolean;
  significant_after_holm: boolean;
}

interface ExperimentSummary {
  experiment_id: string;
  symbol: string;
  status: string;
  created_at: string;
  metrics: MetricsRow[];
  hypothesis_tests: HypothesisTest[];
  dm_comparisons_5d: DMComparison[];
  dm_comparisons_20d: DMComparison[];
}

interface ExperimentListItem {
  experiment_id: string;
  symbol: string;
  created_at: string;
  status: string;
  n_models: number;
}

interface PublishedRow {
  symbol: string;
  horizon: string;
  hv: number;
  hv_ci: [number, number];
  ewma: number;
  ewma_ci: [number, number];
  garch: number;
  garch_ci: [number, number];
  har: number;
  har_ci: [number, number];
  xgboost: number;
  xgboost_ci: [number, number];
  best: string;
}

interface PublishedResults {
  id: string;
  title: string;
  h1_verdict: string;
  h1_statement: string;
  h1_summary: string;
  data_source: string;
  test_period: string;
  n_test: number;
  methodology: string[];
  models: string[];
  rows: PublishedRow[];
  dm_tests: {
    symbol: string;
    horizon: string;
    dm_stat: number;
    p_value: number;
    significant_after_holm: boolean;
  }[];
  shuffle_tests: {
    symbol: string;
    horizon: string;
    real_qlike: number;
    shuffled_qlike: number;
    edge_preserved: boolean;
  }[];
}

const API_BASE = import.meta.env.VITE_API_URL || '';

export function MLResearchStudio() {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentSummary | null>(null);
  const [published, setPublished] = useState<PublishedResults | null>(null);
  const [showPublished, setShowPublished] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // New experiment form
  const [newSymbol, setNewSymbol] = useState('SPY');
  const [newStartDate, setNewStartDate] = useState('2018-01-01');
  const [newSeed, setNewSeed] = useState(42);
  const [submitting, setSubmitting] = useState(false);

  const loadExperiments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/experiments`);
      if (!res.ok) throw new Error('Failed to load experiments');
      const data = await res.json();
      setExperiments(data);
    } catch (e) {
      console.error('Failed to load experiments:', e);
    }
  }, []);

  useEffect(() => {
    loadExperiments();
    fetch(`${API_BASE}/api/v1/experiments/published`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load published results');
        return res.json();
      })
      .then((data: PublishedResults) => {
        setPublished(data);
        setShowPublished(true);
      })
      .catch((e) => {
        console.error('Failed to load published FR-07 results:', e);
      });
  }, [loadExperiments]);

  const loadExperimentSummary = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/experiments/${id}/summary`);
      if (!res.ok) throw new Error('Failed to load experiment');
      const data = await res.json();
      setSelectedExperiment(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const startExperiment = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: newSymbol,
          start_date: newStartDate,
          seed: newSeed,
          n_bootstrap: 500,
        }),
      });
      if (!res.ok) throw new Error('Failed to start experiment');
      const data = await res.json();
      alert(`Experiment started: ${data.experiment_id}`);
      loadExperiments();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start experiment');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <UsageGuide
        storageKey="ml"
        items={[
          'The FR-07 committed results table is preloaded. Symbol, Start Date, and Random Seed plus Run Experiment start a new walk-forward job (slow on a small host). Refresh List reloads past runs.',
          'HV is trailing historical vol. EWMA is RiskMetrics-style decay (λ=0.94). GARCH(1,1) is conditional variance. HAR-RV uses daily/weekly/monthly realized-vol lags. XGBoost is gradient-boosted trees on those features.',
          'QLIKE is the scoring rule — lower is better. Bold blue in the table is the best model on that row.',
          'Bracketed numbers are 95% bootstrap confidence intervals. Overlapping CIs mean the ranking is noisy.',
          'DM is Diebold-Mariano of XGBoost vs HAR-RV. Holm is a multiple-testing correction — ✓ means still significant after Holm.',
          'The shuffle test permutes targets; if shuffled QLIKE beats the real fit, treat that row as leakage/noise (NVDA 20d).',
          'Headline: H1 (XGBoost beats HAR-RV) is NOT SUPPORTED after Holm correction.',
        ]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          ML Research Studio
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Run and analyze volatility forecasting experiments with walk-forward validation,
          bootstrap confidence intervals, and Diebold-Mariano hypothesis testing.
        </p>

        {/* New Experiment Form */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
          <h3 className="font-semibold mb-3 text-gray-900 dark:text-white">
            Start New Experiment
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">
                Symbol
              </label>
              <select
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-600 dark:border-gray-500 dark:text-white"
              >
                <option value="SPY">SPY</option>
                <option value="QQQ">QQQ</option>
                <option value="NVDA">NVDA</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">
                Start Date
              </label>
              <input
                type="date"
                value={newStartDate}
                onChange={(e) => setNewStartDate(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-600 dark:border-gray-500 dark:text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">
                Random Seed
              </label>
              <input
                type="number"
                value={newSeed}
                onChange={(e) => setNewSeed(parseInt(e.target.value))}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-600 dark:border-gray-500 dark:text-white"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={startExperiment}
                disabled={submitting}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {submitting ? 'Starting...' : 'Run Experiment'}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        {/* Experiments List */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1">
            <h3 className="font-semibold mb-3 text-gray-900 dark:text-white">
              Experiments
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {published && (
                <button
                  onClick={() => {
                    setShowPublished(true);
                    setSelectedExperiment(null);
                  }}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    showPublished
                      ? 'bg-blue-50 border-blue-500 dark:bg-blue-900'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-700 border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">
                    FR-07 committed results
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    SPY · QQQ · NVDA · 252 test days
                  </div>
                  <div className="text-xs mt-1 text-green-600">preloaded</div>
                </button>
              )}
              {experiments.length === 0 && !published ? (
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  No experiments yet. Run one above.
                </p>
              ) : experiments.length === 0 ? (
                <p className="text-gray-500 dark:text-gray-400 text-xs">
                  New runs appear here. The table on the right is the committed README result.
                </p>
              ) : (
                experiments.map((exp) => (
                  <button
                    key={exp.experiment_id}
                    onClick={() => {
                      setShowPublished(false);
                      loadExperimentSummary(exp.experiment_id);
                    }}
                    className={`w-full text-left p-3 rounded-lg border transition-colors ${
                      selectedExperiment?.experiment_id === exp.experiment_id
                        ? 'bg-blue-50 border-blue-500 dark:bg-blue-900'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-700 border-gray-200 dark:border-gray-600'
                    }`}
                  >
                    <div className="font-medium text-gray-900 dark:text-white">
                      {exp.symbol}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {new Date(exp.created_at).toLocaleDateString()}
                    </div>
                    <div className={`text-xs mt-1 ${
                      exp.status === 'completed' ? 'text-green-600' :
                      exp.status === 'error' ? 'text-red-600' : 'text-yellow-600'
                    }`}>
                      {exp.status}
                    </div>
                  </button>
                ))
              )}
            </div>
            <button
              onClick={loadExperiments}
              className="mt-3 text-sm text-blue-600 hover:text-blue-800"
            >
              Refresh List
            </button>
          </div>

          {/* Experiment Details */}
          <div className="md:col-span-2">
            {loading ? (
              <div className="text-center py-12 text-gray-500">Loading...</div>
            ) : showPublished && published ? (
              <PublishedResultsPanel published={published} />
            ) : selectedExperiment ? (
              <ExperimentDetails experiment={selectedExperiment} />
            ) : (
              <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                Select an experiment to view results
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function fmtCI(value: number, ci: [number, number]) {
  return `${value.toFixed(2)} [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]`;
}

function PublishedResultsPanel({ published }: { published: PublishedResults }) {
  const models = [
    { key: 'hv', label: 'HV' },
    { key: 'ewma', label: 'EWMA' },
    { key: 'garch', label: 'GARCH' },
    { key: 'har', label: 'HAR-RV' },
    { key: 'xgboost', label: 'XGBoost' },
  ] as const;

  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold text-gray-900 dark:text-white">{published.title}</h4>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          {published.data_source}. Test window: {published.test_period} ({published.n_test} observations).
        </p>
      </div>

      <div className={`rounded-lg p-4 ${
        published.h1_verdict === 'NOT SUPPORTED'
          ? 'bg-amber-50 border border-amber-200'
          : 'bg-green-50 border border-green-200'
      }`}>
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm text-gray-800">{published.h1_statement}</span>
          <span className="px-2 py-1 rounded text-xs font-semibold bg-white text-amber-800 border border-amber-300 whitespace-nowrap">
            {published.h1_verdict}
          </span>
        </div>
        <p className="text-sm text-gray-700 mt-2">{published.h1_summary}</p>
      </div>

      <div className="overflow-x-auto border rounded-lg">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-700">
              <th className="px-2 py-2 text-left">Symbol</th>
              <th className="px-2 py-2 text-left">H</th>
              {models.map((m) => (
                <th key={m.key} className="px-2 py-2 text-right">{m.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {published.rows.map((row) => (
              <tr key={`${row.symbol}-${row.horizon}`} className="border-t">
                <td className="px-2 py-2 font-medium">{row.symbol}</td>
                <td className="px-2 py-2">{row.horizon}</td>
                {models.map((m) => {
                  const value = row[m.key] as number;
                  const ci = row[`${m.key}_ci` as keyof PublishedRow] as [number, number];
                  const isBest = row.best === m.key;
                  return (
                    <td
                      key={m.key}
                      className={`px-2 py-2 text-right font-mono ${isBest ? 'font-bold text-blue-700' : ''}`}
                    >
                      {fmtCI(value, ci)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-gray-500 px-2 py-2">
          QLIKE ± 95% CI (lower is better). Bold = best model on that row.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border rounded-lg overflow-hidden">
          <div className="bg-gray-100 dark:bg-gray-700 px-3 py-2 font-medium text-sm">
            XGBoost vs HAR-RV (DM)
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="px-2 py-1 text-left">Case</th>
                <th className="px-2 py-1 text-right">DM</th>
                <th className="px-2 py-1 text-right">p</th>
                <th className="px-2 py-1 text-center">Holm</th>
              </tr>
            </thead>
            <tbody>
              {published.dm_tests.map((t) => (
                <tr key={`${t.symbol}-${t.horizon}`} className="border-t">
                  <td className="px-2 py-1">{t.symbol} {t.horizon}</td>
                  <td className="px-2 py-1 text-right font-mono">{t.dm_stat.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right font-mono">{t.p_value.toFixed(3)}</td>
                  <td className="px-2 py-1 text-center">{t.significant_after_holm ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="border rounded-lg overflow-hidden">
          <div className="bg-gray-100 dark:bg-gray-700 px-3 py-2 font-medium text-sm">
            Shuffle test (leakage check)
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="px-2 py-1 text-left">Case</th>
                <th className="px-2 py-1 text-right">Real</th>
                <th className="px-2 py-1 text-right">Shuffled</th>
                <th className="px-2 py-1 text-center">Edge</th>
              </tr>
            </thead>
            <tbody>
              {published.shuffle_tests.map((t) => (
                <tr key={`${t.symbol}-${t.horizon}`} className="border-t">
                  <td className="px-2 py-1">{t.symbol} {t.horizon}</td>
                  <td className="px-2 py-1 text-right font-mono">{t.real_qlike.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right font-mono">{t.shuffled_qlike.toFixed(2)}</td>
                  <td className="px-2 py-1 text-center">{t.edge_preserved ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-gray-500 px-2 py-2">
            NVDA 20d is the case where shuffled targets beat the real fit — treat that row as noise.
          </p>
        </div>
      </div>
    </div>
  );
}

function ExperimentDetails({ experiment }: { experiment: ExperimentSummary }) {
  const { metrics, hypothesis_tests, dm_comparisons_5d, dm_comparisons_20d } = experiment;

  const metrics5d = metrics.filter(m => m.horizon === '5d');
  const metrics20d = metrics.filter(m => m.horizon === '20d');

  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
          Experiment: {experiment.symbol}
        </h4>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          Created: {new Date(experiment.created_at).toLocaleString()} | 
          Status: <span className={experiment.status === 'completed' ? 'text-green-600' : 'text-red-600'}>
            {experiment.status}
          </span>
        </div>
      </div>

      {/* Metrics Tables */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricsTable title="5-Day Horizon" metrics={metrics5d} />
        <MetricsTable title="20-Day Horizon" metrics={metrics20d} />
      </div>

      {/* Hypothesis Tests */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h5 className="font-semibold mb-3 text-gray-900 dark:text-white">
          Hypothesis Tests
        </h5>
        <div className="space-y-2">
          {hypothesis_tests.map((test, i) => (
            <div key={i} className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {test.hypothesis}
              </span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                test.result === 'SUPPORTED'
                  ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                  : 'bg-gray-100 text-gray-800 dark:bg-gray-600 dark:text-gray-200'
              }`}>
                {test.result}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* DM Comparisons */}
      {(dm_comparisons_5d.length > 0 || dm_comparisons_20d.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DMComparisonTable title="DM Tests (5d)" comparisons={dm_comparisons_5d} />
          <DMComparisonTable title="DM Tests (20d)" comparisons={dm_comparisons_20d} />
        </div>
      )}

      {/* Metrics Visualization */}
      <div>
        <h5 className="font-semibold mb-3 text-gray-900 dark:text-white">
          Model Comparison
        </h5>
        <Plot
          data={[
            {
              x: metrics.map(m => `${m.model} (${m.horizon})`),
              y: metrics.map(m => m.qlike),
              type: 'bar',
              name: 'QLIKE',
              marker: { color: '#3b82f6' },
            },
          ]}
          layout={{
            height: 300,
            margin: { l: 50, r: 20, t: 30, b: 80 },
            xaxis: { title: { text: 'Model' }, tickangle: -45 },
            yaxis: { title: { text: 'QLIKE (lower is better)' } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}

function MetricsTable({ title, metrics }: { title: string; metrics: MetricsRow[] }) {
  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-600 rounded-lg overflow-hidden">
      <div className="bg-gray-100 dark:bg-gray-700 px-4 py-2 font-medium text-gray-900 dark:text-white">
        {title}
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b dark:border-gray-600">
            <th className="px-4 py-2 text-left text-gray-600 dark:text-gray-300">Model</th>
            <th className="px-4 py-2 text-right text-gray-600 dark:text-gray-300">MAE</th>
            <th className="px-4 py-2 text-right text-gray-600 dark:text-gray-300">RMSE</th>
            <th className="px-4 py-2 text-right text-gray-600 dark:text-gray-300">QLIKE</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m, i) => (
            <tr key={i} className="border-b dark:border-gray-700">
              <td className="px-4 py-2 text-gray-900 dark:text-white">{m.model}</td>
              <td className="px-4 py-2 text-right text-gray-700 dark:text-gray-300">
                {m.mae?.toFixed(6) || '-'}
              </td>
              <td className="px-4 py-2 text-right text-gray-700 dark:text-gray-300">
                {m.rmse?.toFixed(6) || '-'}
              </td>
              <td className="px-4 py-2 text-right text-gray-700 dark:text-gray-300">
                {m.qlike?.toFixed(4) || '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DMComparisonTable({ title, comparisons }: { title: string; comparisons: DMComparison[] }) {
  if (comparisons.length === 0) return null;
  
  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-600 rounded-lg overflow-hidden">
      <div className="bg-gray-100 dark:bg-gray-700 px-4 py-2 font-medium text-gray-900 dark:text-white">
        {title}
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b dark:border-gray-600">
            <th className="px-3 py-2 text-left text-gray-600 dark:text-gray-300">Model</th>
            <th className="px-3 py-2 text-right text-gray-600 dark:text-gray-300">DM Stat</th>
            <th className="px-3 py-2 text-right text-gray-600 dark:text-gray-300">p-value</th>
            <th className="px-3 py-2 text-center text-gray-600 dark:text-gray-300">Sig.</th>
          </tr>
        </thead>
        <tbody>
          {comparisons.map((c, i) => (
            <tr key={i} className="border-b dark:border-gray-700">
              <td className="px-3 py-2 text-gray-900 dark:text-white">{c.model}</td>
              <td className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">
                {c.dm_statistic?.toFixed(3) || '-'}
              </td>
              <td className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">
                {c.p_value?.toFixed(4) || '-'}
              </td>
              <td className="px-3 py-2 text-center">
                {c.significant_after_holm ? (
                  <span className="text-green-600">✓</span>
                ) : (
                  <span className="text-gray-400">-</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
