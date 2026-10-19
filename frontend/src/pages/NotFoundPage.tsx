import { Link, useLocation } from 'react-router'
import { paths } from '../paths'

export function NotFoundPage() {
  const { pathname } = useLocation()
  return (
    <div className="mx-auto max-w-2xl px-3 py-5">
      <title>Not found · CardioScreen</title>
      <h1 className="text-2xl font-semibold tracking-tight">No page here</h1>
      <p className="mt-3">There is no screen at {pathname}.</p>
      <Link to={paths.select} className="mt-4 inline-block text-sm font-medium text-accent underline
                                         underline-offset-2 hover:text-accent-strong">
        Screen an ECG
      </Link>
    </div>
  )
}
