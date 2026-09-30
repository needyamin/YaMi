import { createContext, useContext, useCallback, useMemo, type ReactNode } from "react"
import { createElement } from "react"
import en from "./en"

// English only, deliberately. The engine serves a single audience and every
// string in the UI is authored in English; keeping translation dictionaries
// around meant shipping five half-maintained copies of the same text. Adding a
// language back means adding a dict to DICTS and an entry to LOCALES — the
// provider itself needs no changes (it already falls back to `en` per key).
const LOCALES = [
  { code: "en", label: "English" },
] as const

const DICTS: Record<string, Record<string, string>> = { en }

interface LocaleContext {
  locale: string
  localeName: string
  setLocale: (code: string) => void
  t: (key: string, vars?: Record<string, string | number>) => string
  locales: readonly { code: string; label: string }[]
}

function interpolate(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(vars[key] ?? `{{${key}}}`))
}

const Ctx = createContext<LocaleContext>({
  locale: "en",
  localeName: "English",
  setLocale: () => {},
  t: (key) => key,
  locales: LOCALES,
})

export function LocaleProvider({ children }: { children: ReactNode }) {
  // `t` never changes identity now that there is only one dictionary.
  const t = useCallback((key: string, vars?: Record<string, string | number>) => {
    const template = DICTS.en[key] ?? key
    return interpolate(template, vars)
  }, [])

  const value = useMemo(() => ({
    locale: "en",
    localeName: "English",
    setLocale: () => {},
    t,
    locales: LOCALES,
  }), [t])

  return createElement(Ctx.Provider, { value }, children)
}

export function useLocale() {
  return useContext(Ctx)
}
