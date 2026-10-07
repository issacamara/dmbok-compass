import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AuthPanel } from "./AuthPanel";
import "./styles.css";

export function App() {
  return (
    <main id="main-content">
      <p className="eyebrow">DMBOK COMPASS</p>
      <h1>Grounded data management guidance.</h1>
      <p className="intro">Sign in with your verified email to continue.</p>
      <AuthPanel />
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
