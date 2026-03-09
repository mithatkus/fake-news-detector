import LimeVisualization from './LimeVisualization'

export default function ResultCard({ result, modelLabel }) {
  const isReal = result.label === 'Real'
  const pct = (v) => `${(v * 100).toFixed(1)}%`

  return (
    <div
      className={`rounded-2xl border-2 p-6 transition-all
        ${isReal ? 'border-green-400 bg-green-50' : 'border-red-400 bg-red-50'}`}
    >
      {/* Model badge */}
      {modelLabel && (
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
          {modelLabel}
        </p>
      )}

      {/* Verdict */}
      <div className="text-center mb-4">
        <span
          className={`text-3xl font-extrabold tracking-tight
            ${isReal ? 'text-green-600' : 'text-red-600'}`}
        >
          {isReal ? '✅ REAL NEWS' : '🚨 FAKE NEWS'}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="h-2.5 bg-gray-200 rounded-full mb-4 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700
            ${isReal ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: pct(result.confidence) }}
        />
      </div>

      {/* Probability breakdown */}
      <div className="grid grid-cols-3 gap-2 text-center mb-2">
        <div>
          <p className="text-xs text-gray-500">Confidence</p>
          <p className="text-xl font-bold text-gray-800">{pct(result.confidence)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Real prob.</p>
          <p className="text-lg font-semibold text-green-600">{pct(result.real_probability)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Fake prob.</p>
          <p className="text-lg font-semibold text-red-600">{pct(result.fake_probability)}</p>
        </div>
      </div>

      {/* LIME */}
      <LimeVisualization words={result.lime_words} />
    </div>
  )
}
