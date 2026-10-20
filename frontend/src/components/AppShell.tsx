import { useLayoutEffect, useRef } from 'react'
import { Link, NavLink, Outlet } from 'react-router'
import { paths } from '../paths'

const navClass = ({ isActive }: { isActive: boolean }) =>
  `text-sm ${isActive ? 'font-medium text-ink' : 'text-ink-muted hover:text-ink'}`

/**
 * Header, the current screen, and the footer, fixed to the bottom of the window on every screen. The footer's height
 * (measured, as the disclaimer can wrap on a narrow window) is reserved below the content and as scroll padding, so
 * nothing ends up hidden behind it: not the last control on a page, nor one scrolled into view or focused.
 */
export function AppShell() {
  const footer = useRef<HTMLElement>(null)

  useLayoutEffect(() => {
    const el = footer.current
    if (el === null) return
    const root = document.documentElement
    const reserve = () => root.style.setProperty('--footer-height', `${el.getBoundingClientRect().height}px`)
    reserve()
    const observer = new ResizeObserver(reserve)
    observer.observe(el)
    return () => {
      observer.disconnect()
      root.style.removeProperty('--footer-height')
    }
  }, [])

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex h-8 max-w-5xl items-center justify-between px-3">
          <Link to={paths.select} className="font-semibold tracking-tight text-ink">
            CardioScreen
          </Link>
          <nav aria-label="Screens" className="flex gap-3">
            <NavLink to={paths.select} end className={navClass}>
              Screen an ECG
            </NavLink>
            <NavLink to={paths.performance} className={navClass}>
              Model performance
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="flex-1 pb-(--footer-height)">
        <Outlet />
      </main>

      <footer ref={footer} className="fixed inset-x-0 bottom-0 border-t border-line bg-canvas">
        <p className="mx-auto max-w-5xl px-3 py-2 text-sm text-ink">
          Screening aid. Not a diagnostic device. Not for clinical use.
        </p>
      </footer>
    </div>
  )
}
