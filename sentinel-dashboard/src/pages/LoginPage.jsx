import { useMemo, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import './LoginPage.css'

function readErrorMessage(error, fallback) {
  if (!error) return fallback
  if (typeof error.detail === 'string') return error.detail
  if (error.detail?.message) return error.detail.message
  return error.message || fallback
}

export default function LoginPage() {
  const { login, bootstrapAdmin, needsBootstrap } = useAuth()
  const [identifier, setIdentifier] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const title = useMemo(
    () => (needsBootstrap ? 'Initial Operator Setup' : 'Operator Login'),
    [needsBootstrap]
  )

  const submit = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')

    try {
      if (needsBootstrap) {
        await bootstrapAdmin({
          identifier,
          password,
          email: email || undefined,
        })
      }

      await login(identifier, password)
    } catch (err) {
      setError(readErrorMessage(err, 'Authentication failed'))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-panel">
        <div className="auth-brand">SENTINEL</div>
        <h1 className="auth-title">{title}</h1>
        <p className="auth-sub">
          {needsBootstrap
            ? 'No users exist yet. Create the first admin/operator account to unlock enrollment and camera management.'
            : 'Sign in with your username or email to use protected enrollment and management features.'}
        </p>

        <form className="auth-form" onSubmit={submit}>
          <label className="auth-field">
            <span>Username or email</span>
            <input
              value={identifier}
              onChange={(event) => setIdentifier(event.target.value)}
              placeholder="operator or operator@example.com"
              autoComplete="username"
              required
            />
          </label>

          {needsBootstrap && (
            <label className="auth-field">
              <span>Email (optional)</span>
              <input
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="operator@example.com"
                autoComplete="email"
              />
            </label>
          )}

          <label className="auth-field">
            <span>Password</span>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              placeholder="Enter password"
              autoComplete={needsBootstrap ? 'new-password' : 'current-password'}
              required
            />
          </label>

          <button className="auth-submit" type="submit" disabled={submitting || !identifier || !password}>
            {submitting ? 'Processing...' : needsBootstrap ? 'Create First Admin' : 'Login'}
          </button>
        </form>

        {error && <div className="auth-message auth-error">{error}</div>}

        <div className="auth-footer">
          <span>JWT access tokens protect enrollment and state-changing actions.</span>
          <span>New enrollments are attributed to the authenticated operator automatically.</span>
        </div>
      </div>
    </div>
  )
}
