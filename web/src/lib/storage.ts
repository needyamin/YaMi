export interface StringStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

/* Storage keys used to be prefixed "colibri." — the product is YaMi now, so
   they are "yami.". A browser that has been using the dashboard since before
   the rename still holds the old keys, so `migrate` moves them across once.
   Without it a returning user would silently lose their endpoint, model and
   theme. The old key is deleted rather than left behind so the migration is
   idempotent and nothing stale survives to confuse a later rename. */
const PREFIX = "yami"
const LEGACY_PREFIX = "colibri"
const SETTING_KEYS = ["baseUrl", "model", "theme"] as const

export function migrate(storage: StringStorage) {
  try {
    for (const name of SETTING_KEYS) {
      const next = `${PREFIX}.${name}`
      const previous = `${LEGACY_PREFIX}.${name}`
      const value = storage.getItem(previous)
      if (value && !storage.getItem(next)) storage.setItem(next, value)
      // Always drop the old key, even when a newer value already won: leaving it
      // behind would let a stale value reappear if the new one is ever cleared.
      storage.removeItem(previous)
    }
    // The API key has never been persisted on purpose; drop any legacy copy.
    storage.removeItem(`${LEGACY_PREFIX}.apiKey`)
  } catch { /* restricted storage mode */ }
}

export function stored(storage: Pick<StringStorage, "getItem">, key: string, fallback: string) {
  try { return storage.getItem(key) || fallback } catch { return fallback }
}

export function persistPublicSettings(storage: StringStorage, baseUrl: string, model: string) {
  try {
    storage.setItem(`${PREFIX}.baseUrl`, baseUrl)
    storage.setItem(`${PREFIX}.model`, model)
    // API credentials intentionally remain memory-only. Remove values left by
    // older web releases whenever public settings are persisted.
    storage.removeItem(`${LEGACY_PREFIX}.apiKey`)
  } catch { /* restricted storage mode */ }
}
