const MODELS = [
  {
    value: 'tfidf',
    label: 'TF-IDF Logistic Regression',
    badge: 'Fast',
    badgeColor: 'bg-blue-100 text-blue-700',
    description: '~99% accuracy · ~0.5 s response',
  },
  {
    value: 'cnn',
    label: 'CNN + Word2Vec',
    badge: 'More Accurate',
    badgeColor: 'bg-purple-100 text-purple-700',
    description: '~98.6% accuracy · ~5–10 s response',
  },
]

export default function ModelSelector({ value, onChange }) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700">Select model</label>
      <div className="flex gap-3">
        {MODELS.map((m) => (
          <button
            key={m.value}
            type="button"
            onClick={() => onChange(m.value)}
            className={`flex-1 rounded-xl border-2 p-4 text-left transition-all
              ${
                value === m.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
              }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-sm text-gray-800">{m.label}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${m.badgeColor}`}>
                {m.badge}
              </span>
            </div>
            <p className="text-xs text-gray-500">{m.description}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
