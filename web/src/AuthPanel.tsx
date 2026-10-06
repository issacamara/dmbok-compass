import { useEffect, useState, type FormEvent } from "react";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  sendEmailVerification,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signOut,
  type User,
} from "firebase/auth";
import { auth, firebaseConfigured } from "./firebase";

type Mode = "sign-in" | "sign-up" | "reset-password";

const errors: Record<string, string> = {
  "auth/email-already-in-use": "An account already exists for this email. Try signing in.",
  "auth/invalid-credential": "Email or password is incorrect.",
  "auth/invalid-email": "Enter a valid email address.",
  "auth/weak-password": "Choose a stronger password with at least 6 characters.",
  "auth/too-many-requests": "Too many attempts. Wait a little and try again.",
  "auth/network-request-failed": "Connection failed. Check your internet and try again.",
};

function authError(error: unknown): string {
  const code = typeof error === "object" && error !== null && "code" in error
    ? String(error.code)
    : "";
  return errors[code] ?? "Authentication could not be completed. Please try again.";
}

export function AuthPanel() {
  const [mode, setMode] = useState<Mode>("sign-in");
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => onAuthStateChanged(auth, (currentUser) => {
    setUser(currentUser);
    setLoading(false);
  }, () => {
    setError("Could not connect to the identity service. Refresh to try again.");
    setLoading(false);
  }), []);

  if (!firebaseConfigured) {
    return <p role="alert">Firebase Authentication is not configured. Add the Firebase web app settings to `web/.env.local`.</p>;
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    const form = new FormData(event.currentTarget);
    const email = String(form.get("email") ?? "").trim();
    const password = String(form.get("password") ?? "");
    try {
      if (mode === "sign-up") {
        const credential = await createUserWithEmailAndPassword(auth, email, password);
        await sendEmailVerification(credential.user);
        await signOut(auth);
        setMode("sign-in");
        setMessage("Check your inbox for a verification link before signing in.");
      } else if (mode === "reset-password") {
        await sendPasswordResetEmail(auth, email);
        setMessage("If an account exists for that email, a password reset link is on its way.");
      } else {
        const credential = await signInWithEmailAndPassword(auth, email, password);
        if (!credential.user.emailVerified) {
          await signOut(auth);
          setError("Verify your email using the link we sent before signing in.");
        }
      }
    } catch (cause) {
      setError(authError(cause));
    } finally {
      setBusy(false);
    }
  }

  async function resendVerification() {
    if (!user) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await sendEmailVerification(user);
      setMessage("A new verification link has been sent.");
    } catch (cause) {
      setError(authError(cause));
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <p role="status">Checking your sign-in…</p>;

  if (user?.emailVerified) {
    return (
      <section className="auth-card" aria-labelledby="signed-in-title">
        <h2 id="signed-in-title">You’re signed in</h2>
        <p className="intro">Signed in as {user.email}.</p>
        <p role="status">Your account is email verified.</p>
        <button className="secondary" onClick={() => signOut(auth)}>Sign out</button>
      </section>
    );
  }

  if (user) {
    return (
      <section className="auth-card" aria-labelledby="verify-title">
        <h2 id="verify-title">Verify your email</h2>
        <p className="intro">Use the link sent to {user.email} to verify your address before continuing.</p>
        {message && <p role="status">{message}</p>}
        {error && <p className="error" role="alert">{error}</p>}
        <button disabled={busy} onClick={resendVerification}>{busy ? "Sending…" : "Resend verification link"}</button>
        <button className="secondary" onClick={() => signOut(auth)}>Sign out</button>
      </section>
    );
  }

  const heading = mode === "sign-up" ? "Request access" : mode === "reset-password" ? "Reset password" : "Sign in";
  return (
    <section className="auth-card" aria-labelledby="auth-title">
      <h2 id="auth-title">{heading}</h2>
      <p className="intro">Use your email and password to access DMBOK Compass.</p>
      <form onSubmit={submit}>
        <label htmlFor="email">Email</label>
        <input id="email" name="email" type="email" autoComplete="email" required />
        {mode !== "reset-password" && <>
          <label htmlFor="password">Password</label>
          <input id="password" name="password" type="password" autoComplete={mode === "sign-up" ? "new-password" : "current-password"} minLength={6} required />
        </>}
        <button disabled={busy} type="submit">{busy ? "Please wait…" : heading}</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      <div className="auth-links">
        {mode === "sign-in" && <>
          <button className="link" onClick={() => { setMode("reset-password"); setError(""); setMessage(""); }}>Forgot password?</button>
          <button className="link" onClick={() => { setMode("sign-up"); setError(""); setMessage(""); }}>Request access</button>
        </>}
        {mode !== "sign-in" && <button className="link" onClick={() => { setMode("sign-in"); setError(""); setMessage(""); }}>Back to sign in</button>}
      </div>
    </section>
  );
}
