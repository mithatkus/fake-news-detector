import ResultCard from './ResultCard'

const MODEL_LABELS = {
  tfidf: 'TF-IDF Logistic Regression',
  cnn: 'CNN + Word2Vec',
}

/**
 * Side-by-side results for both models.
 * `results` shape: { tfidf: PredictResponse | null, cnn: PredictResponse | null }
 */
export default function ComparisonMode({ results, loading }) {
  const models = ['tfidf', 'cnn']

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {models.map((m) => {
        const r = results[m]
        return (
          <div key={m}>
            <h3 className="text-sm font-semibold text-gray-600 mb-3">{MODEL_LABELS[m]}</h3>

            {loading && !r && (
              <div className="rounded-2xl border-2 border-gray-200 bg-gray-50 p-10 flex items-center justify-center">
                <Spinner />
              </div>
            )}

            {r && <ResultCard result={r} />}
          </div>
        )
      })}

      {/* Agreement / disagreement banner */}
      {results.tfidf && results.cnn && (
        <div className="md:col-span-2">
          <Agreement tfidf={results.tfidf} cnn={results.cnn} />
        </div>
      )}
    </div>
  )
}

function Agreement({ tfidf, cnn }) {
  const agree = tfidf.label === cnn.label
  return (
    <div
      className={`rounded-xl px-5 py-3 text-sm font-medium text-center
        ${agree ? 'bg-blue-50 text-blue-700 border border-blue-200' : 'bg-yellow-50 text-yellow-800 border border-yellow-200'}`}
    >
      {agree
        ? `Both models agree: the article is likely ${tfidf.label}.`
        : `Models disagree — TF-IDF says ${tfidf.label}, CNN says ${cnn.label}. Consider the higher-confidence result.`}
    </div>
  )
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-8 w-8 text-gray-400"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  )
}
