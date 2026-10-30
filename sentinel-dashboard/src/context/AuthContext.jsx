import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import {
  bootstrapAdmin as bootstrapAdminRequest,
  clearAccessToken,
  getAccessToken,
  getAuthSetupState,
  getCurrentUser,
  login as loginRequest,
  logout as logoutRequest,
  setUnauthorizedHandler,
} from '../utils/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [initializing, setInitializing] = useState(true)
  const [needsBootstrap, setNeedsBootstrap] = useState(false)

  useEffect(() => {
    const removeHandler = setUnauthorizedHandler(() => {
      clearAccessToken()
      setUser(null)
    })
    return removeHandler
  }, [])

  useEffect(() => {
    let cancelled = false

    async function restoreSession() {
      try {
        const setup = await getAuthSetupState()
        if (!cancelled) {
          setNeedsBootstrap(Boolean(setup?.needs_bootstrap))
        }
      } catch {
        if (!cancelled) {
          setNeedsBootstrap(false)
        }
      }

      const token = getAccessToken()
      if (!token) {
        if (!cancelled) {
          setInitializing(false)
        }
        return
      }

      try {
        const currentUser = await getCurrentUser()
        if (!cancelled) {
          setUser(currentUser)
        }
      } catch {
        clearAccessToken()
        if (!cancelled) {
          setUser(null)
        }
      } finally {
        if (!cancelled) {
          setInitializing(false)
        }
      }
    }

    restoreSession()
    return () => {
      cancelled = true
    }
  }, [])

  const value = useMemo(
    () => ({
      user,
      initializing,
      isAuthenticated: Boolean(user),
      needsBootstrap,
      async login(identifier, password) {
        const response = await loginRequest(identifier, password)
        setUser(response.user)
        setNeedsBootstrap(false)
        return response
      },
      async bootstrapAdmin(payload) {
        const response = await bootstrapAdminRequest(payload)
        setNeedsBootstrap(false)
        return response
      },
      async logout() {
        try {
          await logoutRequest()
        } catch {
          // Stateless JWT logout still clears local auth state.
        } finally {
          clearAccessToken()
          setUser(null)
        }
      },
    }),
    [initializing, needsBootstrap, user]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
