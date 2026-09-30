import { createRoot } from "react-dom/client"

import App from "./App"
import { ErrorBoundary } from "./ErrorBoundary"
import { LocaleProvider } from "./i18n"
import { migrate } from "./lib/storage"
import "./index.css"
import "./chat-design.css"

// Move any "colibri.*" settings to "yami.*" before App reads them, so an
// existing session keeps its endpoint, model and theme across the rename.
migrate(localStorage)

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary>
    <LocaleProvider>
      <App />
    </LocaleProvider>
  </ErrorBoundary>,
)
