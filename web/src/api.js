async function request(path, options = {}) {
  const accessToken = localStorage.getItem('fire-review-token')
  const headers = { ...(options.headers || {}) }
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`
  const response = await fetch(path, { ...options, headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`)
  return data
}

export const api = {
  health: () => request('/api/health'),
  statusMetrics: () => request('/api/status/metrics'),
  library: () => request('/api/library'),
  versionTopics: () => request('/api/policy/version-compare'),
  versionTopic: (topicId) => request(`/api/policy/version-compare/${encodeURIComponent(topicId)}`),
  versionResolve: (payload) => request('/api/policy/version-compare/resolve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  ask: (payload, signal) => request('/api/qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }),
  parseIfc: (file) => {
    const body = new FormData()
    body.append('file', file)
    return request('/api/ifc/parse', { method: 'POST', body })
  },
  askIfcModel: (payload, signal) => request('/api/ifc/model-question', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }),
  prepareIfcReview: (payload, signal) => request('/api/ifc/review-preparation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }),
  askIfcCompliance: (payload, signal) => request('/api/ifc/compliance-question', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }),
}
