const SOURCE_OPTIONS = [
  { value: 'default', label: 'Built-in' },
  { value: 'uploaded', label: 'Uploaded' },
  { value: 'combined', label: 'Combined' },
]

const ANSWER_OPTIONS = [
  { value: 'strict', label: 'Strict RAG' },
  { value: 'hybrid', label: 'Hybrid' },
]

const PRESENTATION_OPTIONS = [
  { value: 'standard', label: 'Standard' },
  { value: 'exam', label: 'Exam' },
]

const EXAM_OPTIONS = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'two_mark', label: '2-Mark' },
  { value: 'five_mark', label: '5-Mark' },
  { value: 'six_mark', label: '6-Mark' },
  { value: 'seven_mark', label: '7-Mark' },
  { value: 'ten_mark', label: '10-Mark' },
  { value: 'comparison', label: 'Comparison' },
  { value: 'viva', label: 'Viva' },
]

function ToggleGroup({ title, options, value, onChange, disabledOptions = [] }) {
  return (
    <section className="panel-section">
      <div className="section-label">{title}</div>
      <div className="toggle-grid">
        {options.map((option) => {
          const disabled = disabledOptions.includes(option.value)
          return (
            <button
              key={option.value}
              type="button"
              className={`toggle-chip ${value === option.value ? 'active' : ''}`}
              onClick={() => onChange(option.value)}
              disabled={disabled}
            >
              {option.label}
            </button>
          )
        })}
      </div>
    </section>
  )
}

export function ControlPanel({
  sessionId,
  uploads,
  sourceMode,
  setSourceMode,
  answerMode,
  setAnswerMode,
  presentationMode,
  setPresentationMode,
  examProfile,
  setExamProfile,
  onUpload,
  onResetChat,
  onResetAll,
  busy,
  defaultReady,
}) {
  const disabledSourceModes = []
  if (!uploads.length) {
    disabledSourceModes.push('uploaded', 'combined')
  }
  if (!defaultReady) {
    disabledSourceModes.push('default', 'combined')
  }

  return (
    <aside className="control-panel">
      <div className="brand-block">
        <div className="brand-tag">Night Before. Still Useful.</div>
        <h1>ExamNight AI</h1>
        <p>
          React frontend for semester revision with PDF uploads, grounded answers,
          and exam-style response shaping.
        </p>
      </div>

      <section className="panel-section">
        <div className="section-label">Session</div>
        <div className="session-card">{sessionId ? sessionId.slice(0, 10) : 'Creating...'}</div>
      </section>

      <section className="panel-section">
        <div className="section-label">Documents</div>
        <label className="upload-shell">
          <span>Upload PDF Notes</span>
          <input type="file" accept="application/pdf" multiple onChange={onUpload} disabled={busy} />
        </label>
        <div className="file-list">
          {uploads.length ? uploads.map((file) => <span key={file} className="file-pill">{file}</span>) : <span className="muted">No uploaded PDFs yet.</span>}
        </div>
      </section>

      <ToggleGroup
        title="Knowledge Source"
        options={SOURCE_OPTIONS}
        value={sourceMode}
        onChange={setSourceMode}
        disabledOptions={disabledSourceModes}
      />

      <ToggleGroup
        title="Answer Mode"
        options={ANSWER_OPTIONS}
        value={answerMode}
        onChange={setAnswerMode}
      />

      <ToggleGroup
        title="Presentation"
        options={PRESENTATION_OPTIONS}
        value={presentationMode}
        onChange={setPresentationMode}
      />

      {presentationMode === 'exam' ? (
        <section className="panel-section">
          <div className="section-label">Exam Format</div>
          <select className="format-select" value={examProfile} onChange={(event) => setExamProfile(event.target.value)}>
            {EXAM_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </section>
      ) : null}

      <section className="panel-section action-row">
        <button type="button" className="ghost-button" onClick={onResetChat} disabled={busy}>Clear Chat</button>
        <button type="button" className="ghost-button danger" onClick={onResetAll} disabled={busy}>Reset All</button>
      </section>
    </aside>
  )
}
