import { useEffect, useState, type FormEvent } from "react";
import { getConfiguration, listUsers, updateConfiguration, updateUser } from "./api/admin";
import type { ApprovalState, QuotaPolicy, UserProfile } from "./api/contracts";

type AdminPanelProps = { token: string; onSignOut: () => void };

const filters: Array<{ value: "" | ApprovalState; label: string }> = [
  { value: "", label: "All users" }, { value: "pending", label: "Pending" },
  { value: "approved", label: "Approved" }, { value: "rejected", label: "Rejected" },
  { value: "deactivated", label: "Deactivated" },
];

const actions: Record<ApprovalState, Array<{ state: ApprovalState; label: string }>> = {
  pending: [{ state: "approved", label: "Approve" }, { state: "rejected", label: "Reject" }],
  approved: [{ state: "deactivated", label: "Deactivate" }], rejected: [], deactivated: [],
};

export function AdminPanel({ token, onSignOut }: AdminPanelProps) {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [policy, setPolicy] = useState<QuotaPolicy | null>(null);
  const [filter, setFilter] = useState<"" | ApprovalState>("pending");
  const [loading, setLoading] = useState(true);
  const [busyUser, setBusyUser] = useState<string | null>(null);
  const [savingPolicy, setSavingPolicy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  async function load(filterValue = filter) {
    setLoading(true); setError("");
    try {
      const [loadedUsers, loadedPolicy] = await Promise.all([
        listUsers(token, filterValue || undefined), policy ? Promise.resolve(policy) : getConfiguration(token),
      ]);
      setUsers(loadedUsers); if (!policy) setPolicy(loadedPolicy);
    } catch { setError("Could not load administrator controls. Try again shortly."); }
    finally { setLoading(false); }
  }

  useEffect(() => { void load(); }, []);

  async function changeUser(userId: string, state: ApprovalState) {
    setBusyUser(userId); setError(""); setMessage("");
    try { await updateUser(token, userId, state); setMessage("User access updated."); await load(); }
    catch { setError("Could not update that user’s access."); }
    finally { setBusyUser(null); }
  }

  async function savePolicy(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!policy) return;
    setSavingPolicy(true); setError(""); setMessage("");
    try { setPolicy(await updateConfiguration(token, policy)); setMessage("Quota limits saved."); }
    catch { setError("Could not save quota limits."); }
    finally { setSavingPolicy(false); }
  }

  return (
    <section className="admin-panel" aria-labelledby="admin-title">
      <div className="admin-header"><div>
        <p className="eyebrow">ADMINISTRATOR WORKSPACE</p>
        <h2 id="admin-title">Access and quota controls</h2>
        <p className="intro">Review registration requests and manage daily request limits.</p>
      </div><button className="secondary" type="button" onClick={onSignOut}>Sign out</button></div>
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p className="status" role="status">{message}</p>}
      <div className="admin-grid">
        <section className="admin-card" aria-labelledby="users-title">
          <div className="section-heading"><h3 id="users-title">Users</h3>
            <label>Filter users<select aria-label="Filter users" value={filter} onChange={(event) => {
              const value = event.target.value as "" | ApprovalState; setFilter(value); void load(value);
            }}>{filters.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
          </div>
          {loading ? <p role="status">Loading users…</p> : users.length === 0 ? <p>No users match this filter.</p> : <ul className="user-list">
            {users.map((user) => <li key={user.user_id}>
              <div><strong>{user.username}</strong><span>{user.email}</span></div>
              <span className={`state state-${user.approval_state}`}>{user.approval_state}</span>
              <div className="user-actions">{actions[user.approval_state].map((action) => <button key={action.state}
                className={action.state === "rejected" || action.state === "deactivated" ? "danger" : ""}
                disabled={busyUser === user.user_id} onClick={() => void changeUser(user.user_id, action.state)} type="button">
                {busyUser === user.user_id ? "Saving…" : action.label}
              </button>)}</div>
            </li>)}
          </ul>}
        </section>
        <section className="admin-card" aria-labelledby="quota-title"><h3 id="quota-title">Daily quota limits</h3>
          <p>Limits apply to accepted requests and reset at 00:00 UTC.</p>
          {policy && <form className="quota-form" onSubmit={savePolicy}>
            <label htmlFor="user-limit">Per-user requests</label>
            <input id="user-limit" type="number" min="0" value={policy.per_user_daily_limit} onChange={(event) => setPolicy({ ...policy, per_user_daily_limit: Number(event.target.value) })} />
            <label htmlFor="global-limit">Global requests</label>
            <input id="global-limit" type="number" min="0" value={policy.global_daily_limit} onChange={(event) => setPolicy({ ...policy, global_daily_limit: Number(event.target.value) })} />
            <button disabled={savingPolicy} type="submit">{savingPolicy ? "Saving…" : "Save quota limits"}</button>
          </form>}
        </section>
      </div>
    </section>
  );
}
