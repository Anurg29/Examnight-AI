function normalizeApiBaseUrl(rawValue) {
  const raw = (rawValue || '').trim()
  if (!raw) return ''

  // Keep relative paths (useful for local proxy setups like '/api').
  if (raw.startsWith('/')) {
    return raw.endsWith('/') ? raw.slice(0, -1) : raw
  }

  // If protocol is missing in deployed env, default to https.
  const withProtocol = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`

  try {
    const url = new URL(withProtocol)
    return url.href.endsWith('/') ? url.href.slice(0, -1) : url.href
  } catch {
    throw new Error(
      `Invalid VITE_API_BASE_URL: "${raw}". Use a full URL like https://your-backend.onrender.com`
    )
  }
}

const API_BASE_URL = normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL)

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
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, options)
    return parseResponse(response)
  } catch (error) {
    // Safari often throws: "The string did not match the expected pattern."
    if (error instanceof TypeError || /expected pattern/i.test(String(error?.message || ''))) {
      throw new Error(
        'Cannot reach backend API. Check VITE_API_BASE_URL (must include https://) and ensure backend is running.'
      )
    }
    throw error
  }
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
