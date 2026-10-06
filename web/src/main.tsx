import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

export function App() {
  return (
    <main>
      <p className="eyebrow">DMBOK COMPASS</p>
      <h1>Grounded data management guidance.</h1>
      <p className="intro">The workspace is ready for the secure question-answering experience.</p>
      <span className="status" role="status">API workspace ready</span>
    </main>
  );
}

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
