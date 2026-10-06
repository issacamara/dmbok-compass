import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main";

const authSdk = vi.hoisted(() => ({
  createUserWithEmailAndPassword: vi.fn(),
  onAuthStateChanged: vi.fn(),
  sendEmailVerification: vi.fn(),
  sendPasswordResetEmail: vi.fn(),
  signInWithEmailAndPassword: vi.fn(),
  signOut: vi.fn(),
}));
const identityApi = vi.hoisted(() => ({
  getCurrentProfile: vi.fn(),
  registerProfile: vi.fn(),
  IdentityApiError: class IdentityApiError extends Error {
    constructor(readonly status: number, message: string) { super(message); }
  },
}));

vi.mock("firebase/auth", () => authSdk);
vi.mock("./firebase", () => ({ auth: {}, firebaseConfigured: true }));
vi.mock("./api/identity", () => identityApi);

describe("Firebase email and password identity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    authSdk.onAuthStateChanged.mockImplementation((_auth, callback) => {
      callback(null);
      return vi.fn();
    });
    authSdk.createUserWithEmailAndPassword.mockResolvedValue({
      user: { email: "reader@example.com", getIdToken: vi.fn().mockResolvedValue("firebase-token") },
    });
    authSdk.signInWithEmailAndPassword.mockResolvedValue({ user: { email: "reader@example.com", emailVerified: true } });
    authSdk.sendEmailVerification.mockResolvedValue(undefined);
    authSdk.sendPasswordResetEmail.mockResolvedValue(undefined);
    authSdk.signOut.mockResolvedValue(undefined);
    identityApi.registerProfile.mockResolvedValue({
      user_id: "uid-1", email: "reader@example.com", username: "Reader",
      approval_state: "pending", role: "user", email_verified: false,
    });
    identityApi.getCurrentProfile.mockResolvedValue({
      user_id: "uid-1", email: "reader@example.com", username: "Reader",
      approval_state: "pending", role: "user", email_verified: true,
    });
  });

  it("renders the workspace and email sign-in form", async () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /grounded data management guidance/i })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toHaveAttribute("type", "email");
    expect(screen.getByLabelText("Password")).toHaveAttribute("type", "password");
  });

  it("creates accounts only through Firebase and requires email verification", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Request access" }));
    await user.type(screen.getByLabelText("Username"), "Reader");
    await user.type(screen.getByLabelText("Email"), "reader@example.com");
    await user.type(screen.getByLabelText("Password"), "secure-pass");
    await user.click(screen.getByRole("button", { name: "Request access" }));

    await waitFor(() => expect(authSdk.createUserWithEmailAndPassword).toHaveBeenCalledWith({}, "reader@example.com", "secure-pass"));
    expect(identityApi.registerProfile).toHaveBeenCalledWith("firebase-token", "Reader");
    expect(authSdk.sendEmailVerification).toHaveBeenCalledTimes(1);
    expect(authSdk.sendEmailVerification.mock.calls[0][0].email).toBe("reader@example.com");
    expect(authSdk.signOut).toHaveBeenCalled();
    expect(await screen.findByRole("status")).toHaveTextContent(/verification link/i);
  });

  it("sends a password reset without revealing whether the account exists", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Forgot password?" }));
    await user.type(screen.getByLabelText("Email"), "reader@example.com");
    await user.click(screen.getByRole("button", { name: "Reset password" }));

    expect(authSdk.sendPasswordResetEmail).toHaveBeenCalledWith({}, "reader@example.com");
    expect(await screen.findByRole("status")).toHaveTextContent(/if an account exists/i);
  });

  it("does not allow an unverified account to continue after sign-in", async () => {
    const user = userEvent.setup();
    authSdk.signInWithEmailAndPassword.mockResolvedValue({ user: { email: "reader@example.com", emailVerified: false } });
    render(<App />);
    await user.type(await screen.findByLabelText("Email"), "reader@example.com");
    await user.type(screen.getByLabelText("Password"), "secure-pass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/verify your email/i);
    expect(authSdk.signOut).toHaveBeenCalled();
  });

  it("shows pending approval status from the backend for a verified Firebase user", async () => {
    const signedInUser = {
      email: "reader@example.com",
      emailVerified: true,
      getIdToken: vi.fn().mockResolvedValue("firebase-token"),
    };
    authSdk.onAuthStateChanged.mockImplementation((_auth, callback) => {
      void callback(signedInUser);
      return vi.fn();
    });
    identityApi.getCurrentProfile.mockResolvedValue({
      user_id: "uid-1", email: "reader@example.com", username: "Reader",
      approval_state: "pending", role: "user", email_verified: true,
    });
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Request received" })).toBeInTheDocument();
    expect(identityApi.getCurrentProfile).toHaveBeenCalledWith("firebase-token");
  });

  it("refreshes a pending applicant into the approved journey without requiring a new sign-in", async () => {
    const user = userEvent.setup();
    const signedInUser = {
      email: "reader@example.com",
      emailVerified: true,
      getIdToken: vi.fn().mockResolvedValue("firebase-token"),
    };
    authSdk.onAuthStateChanged.mockImplementation((_auth, callback) => {
      void callback(signedInUser);
      return vi.fn();
    });
    identityApi.getCurrentProfile
      .mockResolvedValueOnce({
        user_id: "uid-1", email: "reader@example.com", username: "Reader",
        approval_state: "pending", role: "user", email_verified: true,
      })
      .mockResolvedValueOnce({
        user_id: "uid-1", email: "reader@example.com", username: "Reader",
        approval_state: "approved", role: "user", email_verified: true,
      });
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Request received" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Refresh access status" }));
    expect(await screen.findByRole("heading", { name: "Access approved" })).toBeInTheDocument();
    expect(identityApi.getCurrentProfile).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["rejected", "Request not approved"],
    ["deactivated", "Account deactivated"],
  ] as const)("explains a %s account state", async (approval_state, heading) => {
    const signedInUser = {
      email: "reader@example.com",
      emailVerified: true,
      getIdToken: vi.fn().mockResolvedValue("firebase-token"),
    };
    authSdk.onAuthStateChanged.mockImplementation((_auth, callback) => {
      void callback(signedInUser);
      return vi.fn();
    });
    identityApi.getCurrentProfile.mockResolvedValue({
      user_id: "uid-1", email: "reader@example.com", username: "Reader",
      approval_state, role: "user", email_verified: true,
    });

    render(<App />);

    expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
  });

  it("lets an authenticated Firebase user complete a missing registration profile", async () => {
    const user = userEvent.setup();
    const signedInUser = {
      email: "reader@example.com",
      emailVerified: true,
      getIdToken: vi.fn().mockResolvedValue("firebase-token"),
    };
    authSdk.onAuthStateChanged.mockImplementation((_auth, callback) => {
      void callback(signedInUser);
      return vi.fn();
    });
    identityApi.getCurrentProfile.mockRejectedValueOnce(new identityApi.IdentityApiError(404, "missing"));
    identityApi.registerProfile.mockResolvedValueOnce({
      user_id: "uid-1", email: "reader@example.com", username: "Reader",
      approval_state: "pending", role: "user", email_verified: true,
    });

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Complete registration" })).toBeInTheDocument();
    await user.type(screen.getByLabelText("Username"), "Reader");
    await user.click(screen.getByRole("button", { name: "Complete registration" }));

    expect(identityApi.registerProfile).toHaveBeenCalledWith("firebase-token", "Reader");
    expect(await screen.findByRole("heading", { name: "Request received" })).toBeInTheDocument();
  });
});
