import { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';

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

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8765';

export function MLResearchStudio() {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentSummary | null>(null);
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
              {experiments.length === 0 ? (
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  No experiments yet. Run one above.
                </p>
              ) : (
                experiments.map((exp) => (
                  <button
                    key={exp.experiment_id}
                    onClick={() => loadExperimentSummary(exp.experiment_id)}
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
            xaxis: { title: 'Model', tickangle: -45 },
            yaxis: { title: 'QLIKE (lower is better)' },
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
