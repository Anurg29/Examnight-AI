import { useEffect, useRef, useState } from 'react'
import { createSession, fetchConfig, resetSession, sendChat, uploadDocuments } from './api/client'
import { ChatMessage } from './components/ChatMessage'
import { ControlPanel } from './components/ControlPanel'

const EMPTY_STATE = {
  sourceMode: 'default',
  answerMode: 'strict',
  presentationMode: 'exam',
  examProfile: 'auto',
}

export default function App() {
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [composer, setComposer] = useState('')
  const [uploads, setUploads] = useState([])
  const [sourceMode, setSourceMode] = useState(EMPTY_STATE.sourceMode)
  const [sourceOptions, setSourceOptions] = useState(['uploaded'])
  const [answerMode, setAnswerMode] = useState(EMPTY_STATE.answerMode)
  const [presentationMode, setPresentationMode] = useState(EMPTY_STATE.presentationMode)
  const [examProfile, setExamProfile] = useState(EMPTY_STATE.examProfile)
  const [defaultReady, setDefaultReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [booting, setBooting] = useState(true)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('Creating your study session...')
  const transcriptRef = useRef(null)

  useEffect(() => {
    async function boot() {
      try {
        const [session, config] = await Promise.all([createSession(), fetchConfig()])
        setSessionId(session.session_id)
        setDefaultReady(config.default_knowledge_base_ready)
        const availableModes = config.source_modes?.length ? config.source_modes : ['uploaded']
        setSourceOptions(availableModes)
        if (!availableModes.includes(sourceMode)) {
          setSourceMode(availableModes[0])
        }
        setInfo(
          config.default_knowledge_base_ready
            ? 'Session ready. Upload PDFs or use the built-in knowledge base.'
            : 'Session ready. Upload PDFs to start asking questions.'
        )
      } catch (bootError) {
        setError(bootError.message)
      } finally {
        setBooting(false)
      }
    }

    boot()
  }, [])

  useEffect(() => {
    if (!sourceOptions.includes(sourceMode)) {
      setSourceMode(sourceOptions[0] || 'uploaded')
      return
    }
    if (!uploads.length && sourceMode === 'combined') {
      setSourceMode(defaultReady ? 'default' : 'uploaded')
    }
    if (!defaultReady && sourceMode === 'default') {
      setSourceMode('uploaded')
    }
  }, [uploads, sourceMode, defaultReady, sourceOptions])

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages, busy])

  const canSend = Boolean(sessionId && composer.trim() && !busy && !booting)

  /** Re-create the session if the server lost it (e.g. after a hot-reload or cold start). */
  async function ensureFreshSession(currentId) {
    const freshSession = await createSession()
    setSessionId(freshSession.session_id)
    setMessages([])
    setUploads([])
    return freshSession.session_id
  }

  async function handleUpload(event) {
    const files = Array.from(event.target.files || [])
    if (!files.length || !sessionId) {
      return
    }

    setBusy(true)
    setError('')
    setInfo('Building your uploaded knowledge base...')
    let activeSessionId = sessionId
    try {
      let response
      try {
        response = await uploadDocuments(activeSessionId, files)
      } catch (firstError) {
        // Session was lost (server restart / hot-reload) — recover silently
        if (firstError.message === 'Session not found.') {
          setInfo('Session expired — reconnecting...')
          activeSessionId = await ensureFreshSession(activeSessionId)
          response = await uploadDocuments(activeSessionId, files)
        } else {
          throw firstError
        }
      }
      setUploads(response.file_names)
      setMessages([])
      setSourceMode(defaultReady ? 'combined' : 'uploaded')
      setInfo(`Processed ${response.pages} pages into ${response.chunks} chunks.`)
    } catch (uploadError) {
      setError(uploadError.message)
    } finally {
      setBusy(false)
      event.target.value = ''
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const query = composer.trim()
    if (!query || !sessionId || busy) {
      return
    }

    if ((sourceMode === 'uploaded' || sourceMode === 'combined') && uploads.length === 0) {
      setError('Upload at least one PDF before asking in uploaded/combined mode.')
      return
    }

    const userMessage = { role: 'user', content: query }
    setComposer('')
    setMessages((current) => [...current, userMessage])
    setBusy(true)
    setError('')
    setInfo('Searching the knowledge base and generating your answer...')

    let activeSessionId = sessionId
    try {
      let response
      try {
        response = await sendChat({
          session_id: activeSessionId,
          query,
          source_mode: sourceMode,
          answer_mode: answerMode,
          presentation_mode: presentationMode,
          exam_profile: examProfile,
        })
      } catch (firstError) {
        if (firstError.message === 'Session not found.') {
          activeSessionId = await ensureFreshSession(activeSessionId)
          response = await sendChat({
            session_id: activeSessionId,
            query,
            source_mode: defaultReady ? 'default' : 'uploaded',
            answer_mode: answerMode,
            presentation_mode: presentationMode,
            exam_profile: examProfile,
          })
        } else {
          throw firstError
        }
      }

      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: response.answer,
          resolvedProfile: response.resolved_profile,
          sources: response.sources,
        },
      ])
      setUploads(response.file_names)
      setInfo('Answer ready. Source cards show where the context came from.')
    } catch (chatError) {
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: chatError.message,
          resolvedProfile: 'error',
          sources: [],
        },
      ])
      setError(chatError.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleReset(clearDocuments) {
    if (!sessionId) {
      return
    }
    setBusy(true)
    setError('')
    try {
      try {
        await resetSession(sessionId, clearDocuments)
      } catch (firstError) {
        // Server lost the session — just boot a fresh one
        if (firstError.message === 'Session not found.') {
          await ensureFreshSession(sessionId)
          setInfo('Session reconnected.')
          return
        }
        throw firstError
      }
      setMessages([])
      setInfo(clearDocuments ? 'Chat and uploaded PDFs cleared.' : 'Chat cleared.')
      if (clearDocuments) {
        setUploads([])
        setSourceMode(defaultReady ? 'default' : 'uploaded')
      }
    } catch (resetError) {
      setError(resetError.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="glow glow-left" />
      <div className="glow glow-right" />

      <ControlPanel
        sessionId={sessionId}
        uploads={uploads}
        sourceMode={sourceMode}
        sourceOptions={sourceOptions}
        setSourceMode={setSourceMode}
        answerMode={answerMode}
        setAnswerMode={setAnswerMode}
        presentationMode={presentationMode}
        setPresentationMode={setPresentationMode}
        examProfile={examProfile}
        setExamProfile={setExamProfile}
        onUpload={handleUpload}
        onResetChat={() => handleReset(false)}
        onResetAll={() => handleReset(true)}
        busy={busy}
        defaultReady={defaultReady}
      />

      <main className="workspace">
        <section className="hero-card">
          <div>
            <div className="hero-label">Interactive Full Stack Revision Workspace</div>
            <h2>Upload semester PDFs, switch answer styles, and get citation-backed exam responses.</h2>
          </div>
          <div className="hero-status">
            <span className={`status-pill ${busy ? 'busy' : 'idle'}`}>{busy ? 'Working' : 'Ready'}</span>
            <span className="status-copy">{info}</span>
          </div>
        </section>

        {error ? <div className="alert-banner">{error}</div> : null}

        <section className="chat-stage">
          <div className="transcript" ref={transcriptRef}>
            {messages.length ? (
              messages.map((message, index) => (
                <ChatMessage key={`${message.role}-${index}-${message.content.slice(0, 24)}`} message={message} />
              ))
            ) : (
              <div className="empty-state">
                <h3>Start with a question or upload your notes.</h3>
                <p>
                  Try prompts like “Explain shock in 7 marks”, “Differentiate anemia and leukemia”,
                  or “Give viva questions on insulin”.
                </p>
              </div>
            )}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={composer}
              onChange={(event) => setComposer(event.target.value)}
              placeholder="Ask your exam question here..."
              rows={4}
            />
            <div className="composer-actions">
              <div className="composer-hint">
                {presentationMode === 'exam' ? 'Exam mode shapes the answer for marks-based responses.' : 'Standard mode gives a normal study answer.'}
              </div>
              <button type="submit" className="send-button" disabled={!canSend}>
                {booting ? 'Starting...' : busy ? 'Thinking...' : 'Send'}
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  )
}
