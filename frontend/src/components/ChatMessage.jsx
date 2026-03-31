const PROFILE_LABELS = {
  standard: 'Standard',
  auto: 'Auto',
  two_mark: '2-Mark',
  five_mark: '5-Mark',
  six_mark: '6-Mark',
  seven_mark: '7-Mark',
  ten_mark: '10-Mark',
  comparison: 'Comparison',
  viva: 'Viva',
  none: 'None',
  error: 'Error',
}

export function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  const profileLabel = PROFILE_LABELS[message.resolvedProfile] || message.resolvedProfile

  return (
    <article className={`message-card ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-heading">
        <span className="message-role">{isUser ? 'You' : 'ExamNight AI'}</span>
        {!isUser && message.resolvedProfile ? (
          <span className="message-profile">{profileLabel}</span>
        ) : null}
      </div>

      <div className="message-body">{message.content}</div>

      {!isUser && message.sources?.length ? (
        <div className="source-stack">
          {message.sources.map((source, index) => (
            <details className="source-card" key={`${source.source}-${index}`}>
              <summary>
                <span>{source.source}</span>
                <span>{source.page ? `Page ${source.page}` : source.kb_label}</span>
              </summary>
              <div className="source-meta">{source.kb_label}</div>
              <div className="source-body">{source.content}</div>
            </details>
          ))}
        </div>
      ) : null}
    </article>
  )
}
