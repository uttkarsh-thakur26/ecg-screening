/** Screen 3. It will show reports/metrics.json as GET /api/metrics serves it; built in a later step. */
export function PerformancePage() {
  return (
    <div className="mx-auto max-w-2xl px-3 py-5">
      <title>Model performance · CardioScreen</title>
      <h1 className="text-2xl font-semibold tracking-tight">Model performance</h1>
      <p className="mt-3 text-ink-muted">
        This screen will show the one-time test-set evaluation (reports/metrics.json, served by the API). It is
        built in a later step.
      </p>
    </div>
  )
}
