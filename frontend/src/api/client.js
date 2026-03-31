const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function parseResponse(response) {
  if (response.ok) {
    return response.json()
  }

  let detail = 'Request failed.'
  try {
    const payload = await response.json()
    detail = payload.detail || detail
  } catch (error) {
    detail = response.statusText || detail
  }
  throw new Error(detail)
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  return parseResponse(response)
}

export function createSession() {
  return request('/api/sessions', { method: 'POST' })
}

export function fetchConfig() {
  return request('/api/config')
}

export function uploadDocuments(sessionId, files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  return request(`/api/sessions/${sessionId}/documents`, {
    method: 'POST',
    body: formData,
  })
}

export function sendChat(payload) {
  return request('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export function resetSession(sessionId, clearDocuments = false) {
  return request(`/api/sessions/${sessionId}/reset?clear_documents=${clearDocuments}`, {
    method: 'POST',
  })
}
