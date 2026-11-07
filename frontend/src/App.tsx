import { useState } from 'react'
import { TrendingUp, Activity, BarChart3, Database, FlaskConical } from 'lucide-react'
import PricingWorkbench from './components/PricingWorkbench'
import VolatilitySurfaceLab from './components/VolatilitySurfaceLab'
import { MLResearchStudio } from './components/MLResearchStudio'

type Tab = 'pricing' | 'surface' | 'ml'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('pricing')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">VolatilityLab</h1>
                <p className="text-sm text-gray-500">Quantitative Options Research Platform</p>
              </div>
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                API Connected
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('pricing')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'pricing'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Activity className="w-4 h-4" />
              Options Pricing Workbench
            </button>
            <button
              onClick={() => setActiveTab('surface')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'surface'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Volatility Surface Lab
            </button>
            <button
              onClick={() => setActiveTab('ml')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'ml'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <FlaskConical className="w-4 h-4" />
              ML Research Studio
            </button>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'pricing' && <PricingWorkbench />}
        {activeTab === 'surface' && <VolatilitySurfaceLab />}
        {activeTab === 'ml' && <MLResearchStudio />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>VolatilityLab v0.1.0 - Portfolio Research Project</span>
            <div className="flex items-center gap-4">
              <span>Models: BSM, CRR Binomial, Monte Carlo GBM</span>
              <span className="flex items-center gap-1">
                <Database className="w-4 h-4" />
                Sample data included
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
