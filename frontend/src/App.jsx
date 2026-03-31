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
        setInfo('Session ready. Upload PDFs or start with the built-in encyclopedia.')
      } catch (bootError) {
        setError(bootError.message)
      } finally {
        setBooting(false)
      }
    }

    boot()
  }, [])

  useEffect(() => {
    if (!uploads.length && sourceMode !== 'default') {
      setSourceMode(defaultReady ? 'default' : 'uploaded')
    }
    if (!defaultReady && sourceMode === 'default' && uploads.length) {
      setSourceMode('uploaded')
    }
  }, [uploads, sourceMode, defaultReady])

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages, busy])

  const canSend = Boolean(sessionId && composer.trim() && !busy && !booting)

  async function handleUpload(event) {
    const files = Array.from(event.target.files || [])
    if (!files.length || !sessionId) {
      return
    }

    setBusy(true)
    setError('')
    setInfo('Building your uploaded knowledge base...')
    try {
      const response = await uploadDocuments(sessionId, files)
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

    const userMessage = { role: 'user', content: query }
    setComposer('')
    setMessages((current) => [...current, userMessage])
    setBusy(true)
    setError('')
    setInfo('Searching the knowledge base and generating your answer...')

    try {
      const response = await sendChat({
        session_id: sessionId,
        query,
        source_mode: sourceMode,
        answer_mode: answerMode,
        presentation_mode: presentationMode,
        exam_profile: examProfile,
      })

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
      await resetSession(sessionId, clearDocuments)
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
