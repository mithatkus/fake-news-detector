import { useState } from 'react'
import { predict } from './api'
import ArticleInput from './components/ArticleInput'
import ModelSelector from './components/ModelSelector'
import ResultCard from './components/ResultCard'
import ComparisonMode from './components/ComparisonMode'

export default function App() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState('single')   // 'single' | 'compare'
  const [model, setModel] = useState('tfidf')

  // Single-model state
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Compare-mode state
  const [compareResults, setCompareResults] = useState({ tfidf: null, cnn: null })
  const [compareLoading, setCompareLoading] = useState(false)

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length
  const isTooShort = text.trim().length > 0 && wordCount < 30

  async function handleSingleAnalyze() {
    if (!text.trim() || isTooShort) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await predict({ text, model })
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleCompare() {
    if (!text.trim() || isTooShort) return
    setCompareLoading(true)
    setError(null)
    setCompareResults({ tfidf: null, cnn: null })

    // Fire both requests concurrently
    const [tfidfPromise, cnnPromise] = [
      predict({ text, model: 'tfidf' }),
      predict({ text, model: 'cnn' }),
    ]

    // Resolve each as it arrives so the UI updates progressively
    tfidfPromise
      .then((r) => setCompareResults((prev) => ({ ...prev, tfidf: r })))
      .catch((e) => setError(e.message))

    cnnPromise
      .then((r) => setCompareResults((prev) => ({ ...prev, cnn: r })))
      .catch((e) => setError(e.message))

    await Promise.allSettled([tfidfPromise, cnnPromise])
    setCompareLoading(false)
  }

  const isCompare = mode === 'compare'

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-extrabold text-gray-900 tracking-tight">
              Fake News Detector
            </h1>
            <p className="text-sm text-gray-500 mt-0.5">
              ML-powered classification with LIME explanations
            </p>
          </div>
          <a
            href="https://github.com/mithatkus/fake-news-detector"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-blue-600 hover:underline"
          >
            GitHub
          </a>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-6">
        {/* Mode toggle */}
        <div className="flex gap-1 bg-gray-100 rounded-xl p-1 w-fit">
          {['single', 'compare'].map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m)
                setResult(null)
                setCompareResults({ tfidf: null, cnn: null })
                setError(null)
              }}
              className={`px-5 py-2 rounded-lg text-sm font-medium transition-all
                ${mode === m ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
            >
              {m === 'single' ? 'Single Model' : 'Compare Both'}
            </button>
          ))}
        </div>

        {/* Input card */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-5">
          {!isCompare && (
            <ModelSelector value={model} onChange={setModel} />
          )}

          <ArticleInput value={text} onChange={setText} />

          <button
            type="button"
            onClick={isCompare ? handleCompare : handleSingleAnalyze}
            disabled={!text.trim() || isTooShort || loading || compareLoading}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300
                       text-white font-semibold transition-colors"
          >
            {loading || compareLoading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                </svg>
                Analyzing…
              </span>
            ) : isCompare ? (
              'Compare Both Models'
            ) : (
              'Analyze Article'
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-300 text-red-700 rounded-xl px-5 py-3 text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        {!isCompare && result && (
          <ResultCard
            result={result}
            modelLabel={model === 'tfidf' ? 'TF-IDF Logistic Regression' : 'CNN + Word2Vec'}
          />
        )}

        {isCompare && (compareLoading || compareResults.tfidf || compareResults.cnn) && (
          <ComparisonMode results={compareResults} loading={compareLoading} />
        )}
      </main>

      {/* Footer */}
      <footer className="text-center text-xs text-gray-400 py-8">
        Trained on the ISOT Fake News Dataset · Ahmed, Traore &amp; Saad (2017)
      </footer>
    </div>
  )
}
