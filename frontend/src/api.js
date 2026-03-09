const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Call the /predict endpoint.
 * @param {{ text: string, model: 'tfidf' | 'cnn' }} params
 * @returns {Promise<PredictResponse>}
 */
export async function predict({ text, model }) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model }),
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed: ${response.status}`)
  }

  return response.json()
}
