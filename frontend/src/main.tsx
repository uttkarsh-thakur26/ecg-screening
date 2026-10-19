// Inter is bundled from node_modules (latin subset, the three weights the design uses); no font is fetched from a CDN.
import '@fontsource/inter/latin-400.css'
import '@fontsource/inter/latin-500.css'
import '@fontsource/inter/latin-600.css'
import './index.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router/dom'
import { router } from './router'

const root = document.getElementById('root')
if (root === null) throw new Error('index.html has no element with id "root"')

createRoot(root).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
