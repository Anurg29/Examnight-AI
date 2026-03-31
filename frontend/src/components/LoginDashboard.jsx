import React, { useState } from 'react';

const API = import.meta.env.VITE_API_BASE_URL || '';

async function apiAuth(path, body) {
  const res = await fetch(`${API}/api/auth/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

export default function LoginDashboard({ onLogin }) {
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
    if (!username.trim() || !password) {
      setError('Please enter both username and password.');
      return;
    }
    setLoading(true);
    try {
      if (mode === 'register') {
        await apiAuth('register', { username: username.trim(), password });
        setSuccessMsg('Account created! You can now log in.');
        setMode('login');
        setPassword('');
      } else {
        const data = await apiAuth('login', { username: username.trim(), password });
        // Store JWT in sessionStorage (cleared when tab closes)
        sessionStorage.setItem('en_token', data.access_token);
        sessionStorage.setItem('en_user', data.username);
        onLogin(data.username);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0d1117 0%, #161b26 60%, #1a2236 100%)',
    }}>
      <div style={{
        width: 360,
        padding: '36px 32px',
        borderRadius: 16,
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 8px 40px rgba(0,0,0,0.5)',
        backdropFilter: 'blur(10px)',
      }}>
        {/* Logo / brand */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{ fontSize: 36, marginBottom: 4 }}>📚</div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#e6edf3', letterSpacing: '-0.5px' }}>
            ExamNight AI
          </h1>
          <p style={{ margin: '4px 0 0', color: '#7d8590', fontSize: 13 }}>
            {mode === 'login' ? 'Sign in to your account' : 'Create a new account'}
          </p>
        </div>

        {/* Toggle pills */}
        <div style={{
          display: 'flex',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: 8,
          padding: 4,
          marginBottom: 24,
          gap: 4,
        }}>
          {['login', 'register'].map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => { setMode(m); setError(''); setSuccessMsg(''); }}
              style={{
                flex: 1,
                padding: '8px 0',
                borderRadius: 6,
                border: 'none',
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: 600,
                transition: 'all 0.2s',
                background: mode === m ? '#2563eb' : 'transparent',
                color: mode === m ? '#fff' : '#7d8590',
              }}
            >
              {m === 'login' ? 'Sign In' : 'Register'}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} noValidate>
          <div style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', fontSize: 12, color: '#8b949e', marginBottom: 6, fontWeight: 500 }}>
              USERNAME
            </label>
            <input
              id="auth-username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="your_username"
              disabled={loading}
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid rgba(255,255,255,0.1)',
                background: 'rgba(255,255,255,0.06)',
                color: '#e6edf3',
                fontSize: 14,
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#2563eb')}
              onBlur={(e) => (e.target.style.borderColor = 'rgba(255,255,255,0.1)')}
            />
          </div>

          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', fontSize: 12, color: '#8b949e', marginBottom: 6, fontWeight: 500 }}>
              PASSWORD
            </label>
            <input
              id="auth-password"
              type="password"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              disabled={loading}
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid rgba(255,255,255,0.1)',
                background: 'rgba(255,255,255,0.06)',
                color: '#e6edf3',
                fontSize: 14,
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#2563eb')}
              onBlur={(e) => (e.target.style.borderColor = 'rgba(255,255,255,0.1)')}
            />
          </div>

          {error && (
            <div style={{
              padding: '10px 12px',
              borderRadius: 8,
              background: 'rgba(239,68,68,0.12)',
              border: '1px solid rgba(239,68,68,0.3)',
              color: '#f87171',
              fontSize: 13,
              marginBottom: 16,
            }}>
              {error}
            </div>
          )}

          {successMsg && (
            <div style={{
              padding: '10px 12px',
              borderRadius: 8,
              background: 'rgba(34,197,94,0.12)',
              border: '1px solid rgba(34,197,94,0.3)',
              color: '#4ade80',
              fontSize: 13,
              marginBottom: 16,
            }}>
              {successMsg}
            </div>
          )}

          <button
            id="auth-submit"
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '11px 0',
              borderRadius: 8,
              border: 'none',
              cursor: loading ? 'not-allowed' : 'pointer',
              background: loading ? '#1e3a8a' : 'linear-gradient(135deg, #2563eb, #1d4ed8)',
              color: '#fff',
              fontSize: 14,
              fontWeight: 700,
              letterSpacing: '0.3px',
              transition: 'opacity 0.2s',
              opacity: loading ? 0.7 : 1,
              boxShadow: loading ? 'none' : '0 4px 12px rgba(37,99,235,0.4)',
            }}
          >
            {loading
              ? mode === 'login' ? 'Signing in…' : 'Creating account…'
              : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  );
}
