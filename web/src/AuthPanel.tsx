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
import { getCurrentProfile, IdentityApiError, registerProfile } from "./api/identity";
import type { UserProfile } from "./api/contracts";
import { auth, firebaseConfigured } from "./firebase";

type Mode = "sign-in" | "sign-up" | "reset-password" | "complete-profile";

const errors: Record<string, string> = {
  "auth/email-already-in-use": "An account already exists for this email. Try signing in.",
  "auth/invalid-credential": "Email or password is incorrect.",
  "auth/user-not-found": "Email or password is incorrect.",
  "auth/wrong-password": "Email or password is incorrect.",
  "auth/user-disabled": "This Firebase account has been disabled. Contact the administrator.",
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
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [profileMissing, setProfileMissing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const activeMode = user?.emailVerified && profileMissing ? "complete-profile" : mode;

  async function refreshProfile(currentUser: User): Promise<void> {
    setLoading(true);
    setError("");
    try {
      setProfile(await getCurrentProfile(await currentUser.getIdToken()));
      setProfileMissing(false);
    } catch (cause) {
      if (cause instanceof IdentityApiError && cause.status === 404) {
        setProfile(null);
        setProfileMissing(true);
      } else {
        setError("Could not load your account access status. Try again shortly.");
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let active = true;
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser?.emailVerified) setLoading(true);
      setUser(currentUser);
      setProfile(null);
      setProfileMissing(false);
      if (currentUser?.emailVerified) {
        try {
          const currentProfile = await getCurrentProfile(await currentUser.getIdToken());
          if (active) setProfile(currentProfile);
        } catch (cause) {
          if (active && cause instanceof IdentityApiError && cause.status === 404) {
            setProfileMissing(true);
          } else if (active) {
            setError("Could not load your account access status. Try again shortly.");
          }
        }
      }
      if (active) setLoading(false);
    }, () => {
      if (active) {
        setError("Could not connect to the identity service. Refresh to try again.");
        setLoading(false);
      }
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

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
    const username = String(form.get("username") ?? "").trim();
    try {
      if (activeMode === "complete-profile") {
        if (!user) throw new Error("Sign in to finish your registration.");
        setProfile(await registerProfile(await user.getIdToken(), username));
        setProfileMissing(false);
        setMessage("Your registration request is complete.");
      } else if (mode === "sign-up") {
        const credential = await createUserWithEmailAndPassword(auth, email, password);
        await registerProfile(await credential.user.getIdToken(), username);
        await sendEmailVerification(credential.user);
        await signOut(auth);
        setMode("sign-in");
        setMessage("Check your inbox for a verification link. Your request will wait for administrator approval.");
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
      setError(cause instanceof IdentityApiError
        ? "Your Firebase account is ready, but the registration service could not save your request. Sign in again to retry."
        : authError(cause));
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

  async function logout() {
    setError("");
    try {
      await signOut(auth);
      setProfile(null);
      setMessage("");
    } catch (cause) {
      setError(authError(cause));
    }
  }

  if (loading) return <p role="status">Checking your account and access…</p>;

  if (user?.emailVerified && profile) {
    const stateContent = {
      pending: ["Request received", "Your email is verified. An administrator must approve your account before you can use DMBOK Compass."],
      approved: ["Access approved", "Your account is verified and approved."],
      rejected: ["Request not approved", "An administrator did not approve this account. Contact the DMBOK Compass administrator if you need help."],
      deactivated: ["Account deactivated", "This account no longer has access. Contact the DMBOK Compass administrator if you need help."],
    } as const;
    const [title, description] = stateContent[profile.approval_state];
    return (
      <section className="auth-card" aria-labelledby="account-title">
        <h2 id="account-title">{title}</h2>
        <p className="intro">{description}</p>
        <p>Signed in as {profile.email}.</p>
        {error && <p className="error" role="alert">{error}</p>}
        <button disabled={loading} onClick={() => refreshProfile(user)}>
          {loading ? "Checking…" : "Refresh access status"}
        </button>
        <button className="secondary" onClick={logout} type="button">Sign out</button>
      </section>
    );
  }

  if (user?.emailVerified && !profile && !profileMissing) {
    return (
      <section className="auth-card" aria-labelledby="access-check-title">
        <h2 id="access-check-title">Checking account access</h2>
        <p className="intro">Your sign-in succeeded, but the registration service did not return your account status.</p>
        {error && <p className="error" role="alert">{error}</p>}
        <button onClick={() => refreshProfile(user)} disabled={loading}>
          {loading ? "Checking…" : "Retry account check"}
        </button>
        <button className="secondary" onClick={logout} type="button">Sign out</button>
      </section>
    );
  }

  if (user && !user.emailVerified) {
    return (
      <section className="auth-card" aria-labelledby="verify-title">
        <h2 id="verify-title">Verify your email</h2>
        <p className="intro">Use the link sent to {user.email} to verify your address before continuing.</p>
        {message && <p role="status">{message}</p>}
        {error && <p className="error" role="alert">{error}</p>}
        <button disabled={busy} onClick={resendVerification} type="button">{busy ? "Sending…" : "Resend verification link"}</button>
        <button className="secondary" onClick={logout} type="button">Sign out</button>
      </section>
    );
  }

  const heading = activeMode === "sign-up" ? "Request access"
    : activeMode === "reset-password" ? "Reset password"
      : activeMode === "complete-profile" ? "Complete registration" : "Sign in";

  return (
    <section className="auth-card" aria-labelledby="auth-title">
      <h2 id="auth-title">{heading}</h2>
      <p className="intro">{activeMode === "complete-profile"
        ? "Add a username to submit your registration request for administrator approval."
        : "Use your email and password to access DMBOK Compass."}</p>
      <form onSubmit={submit}>
        {(activeMode === "sign-up" || activeMode === "complete-profile") && <>
          <label htmlFor="username">Username</label>
          <input id="username" name="username" autoComplete="nickname" maxLength={80} required />
        </>}
        {activeMode !== "complete-profile" && <>
          <label htmlFor="email">Email</label>
          <input id="email" name="email" type="email" autoComplete="email" required />
        </>}
        {activeMode !== "reset-password" && activeMode !== "complete-profile" && <>
          <label htmlFor="password">Password</label>
          <input id="password" name="password" type="password" autoComplete={activeMode === "sign-up" ? "new-password" : "current-password"} minLength={6} required />
        </>}
        <button disabled={busy} type="submit">{busy ? "Please wait…" : heading}</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {!user && <div className="auth-links">
        {activeMode === "sign-in" && <>
          <button className="link" onClick={() => { setMode("reset-password"); setError(""); setMessage(""); }} type="button">Forgot password?</button>
          <button className="link" onClick={() => { setMode("sign-up"); setError(""); setMessage(""); }} type="button">Request access</button>
        </>}
        {activeMode !== "sign-in" && <button className="link" onClick={() => { setMode("sign-in"); setError(""); setMessage(""); }} type="button">Back to sign in</button>}
      </div>}
    </section>
  );
}
