/**
 * Renders LIME word attributions as coloured chips.
 * Green = pushed toward Real, Red = pushed toward Fake.
 * Opacity scales with the absolute weight value.
 */
export default function LimeVisualization({ words }) {
  if (!words || words.length === 0) return null

  const maxAbs = Math.max(...words.map((w) => Math.abs(w.weight)), 0.001)

  return (
    <div className="mt-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-1">
        Key words influencing the prediction
      </h3>
      <p className="text-xs text-gray-400 mb-3">
        <span className="inline-block w-3 h-3 rounded-sm bg-green-500 mr-1 align-middle" />
        pushed toward Real &nbsp;|&nbsp;
        <span className="inline-block w-3 h-3 rounded-sm bg-red-500 mr-1 align-middle" />
        pushed toward Fake
      </p>

      <div className="flex flex-wrap gap-2">
        {words.map(({ word, weight }, i) => {
          const isReal = weight > 0
          const opacity = Math.min(1, Math.abs(weight) / maxAbs * 0.8 + 0.2)
          const bg = isReal ? `rgba(34,197,94,${opacity})` : `rgba(239,68,68,${opacity})`
          const textColor = opacity > 0.5 ? 'text-white' : isReal ? 'text-green-800' : 'text-red-800'

          return (
            <span
              key={i}
              title={`weight: ${weight.toFixed(4)}`}
              className={`px-3 py-1 rounded-full text-sm font-medium ${textColor} cursor-default`}
              style={{ backgroundColor: bg }}
            >
              {word}
            </span>
          )
        })}
      </div>
    </div>
  )
}
