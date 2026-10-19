// The app's routes: Select (screen 1), Result (screen 2) and Performance (screen 3).
export const paths = {
  select: '/',
  recordResult: (id: string) => `/result/record/${encodeURIComponent(id)}`,
  recordResultPattern: '/result/record/:recordId',
  uploadResult: '/result/upload',
  performance: '/performance',
} as const
