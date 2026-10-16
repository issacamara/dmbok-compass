import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { AuthPanel } from "./AuthPanel";
import "./styles.css";

export function App() {
  const [authenticated, setAuthenticated] = useState(false);

  return (
    <main id="main-content">
      {!authenticated && <>
        <p className="eyebrow">DMBOK COMPASS</p>
        <h1>Welcome to DMBOK Compass.</h1>
        <p className="intro">Sign in with your verified email to continue.</p>
      </>}
      <AuthPanel onAuthenticationChange={setAuthenticated} />
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
