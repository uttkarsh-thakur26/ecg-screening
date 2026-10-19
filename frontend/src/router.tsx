import { createBrowserRouter } from 'react-router'
import { AppShell } from './components/AppShell'
import { NotFoundPage } from './pages/NotFoundPage'
import { PerformancePage } from './pages/PerformancePage'
import { ResultPage } from './pages/ResultPage'
import { SelectPage } from './pages/SelectPage'
import { paths } from './paths'

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: paths.select, element: <SelectPage /> },
      { path: paths.recordResultPattern, element: <ResultPage source="test_record" /> },
      { path: paths.uploadResult, element: <ResultPage source="upload" /> },
      { path: paths.performance, element: <PerformancePage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])
