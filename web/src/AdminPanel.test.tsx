import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AdminPanel } from "./AdminPanel";

const adminApi = vi.hoisted(() => ({
  getConfiguration: vi.fn(), listAggregateMetrics: vi.fn(), listUsers: vi.fn(), updateConfiguration: vi.fn(), updateUser: vi.fn(),
}));
vi.mock("./api/admin", () => adminApi);

const pendingUser = {
  user_id: "user-1", email: "reader@example.com", username: "Reader",
  approval_state: "pending" as const, role: "user" as const, email_verified: true,
};
const approvedUser = { ...pendingUser, user_id: "user-2", username: "Approved", approval_state: "approved" as const };

describe("AdminPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    adminApi.listUsers.mockResolvedValue([pendingUser]);
    adminApi.listAggregateMetrics.mockResolvedValue([
      { metric_name: "request_count", numerator: 12, denominator: 20, percentage: 60 },
    ]);
    adminApi.getConfiguration.mockResolvedValue({ per_user_daily_limit: 100, global_daily_limit: 100 });
    adminApi.updateUser.mockResolvedValue({ ...pendingUser, approval_state: "approved" });
    adminApi.updateConfiguration.mockImplementation(async (_token, policy) => policy);
  });

  it("loads users and quota controls, then saves an approval decision and limits", async () => {
    const user = userEvent.setup();
    render(<AdminPanel token="firebase-token" onSignOut={vi.fn()} />);

    expect(await screen.findByText("Reader")).toBeInTheDocument();
    expect(screen.getByLabelText("Per-user requests")).toHaveValue(100);
    await user.click(screen.getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(adminApi.updateUser).toHaveBeenCalledWith("firebase-token", "user-1", "approved"));

    await user.clear(screen.getByLabelText("Per-user requests"));
    await user.type(screen.getByLabelText("Per-user requests"), "20");
    await user.click(screen.getByRole("button", { name: "Save quota limits" }));
    await waitFor(() => expect(adminApi.updateConfiguration).toHaveBeenCalledWith("firebase-token", {
      per_user_daily_limit: 20, global_daily_limit: 100,
    }));
    expect(screen.getByRole("status")).toHaveTextContent("Quota limits saved.");
  });

  it("reports service failures without claiming an update succeeded", async () => {
    adminApi.listUsers.mockRejectedValue(new Error("offline"));
    render(<AdminPanel token="firebase-token" onSignOut={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not load administrator controls/i);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it.each([
    ["Reject", pendingUser, "rejected"],
    ["Deactivate", approvedUser, "deactivated"],
  ] as const)("supports the %s access action", async (label, user, nextState) => {
    const actor = userEvent.setup();
    adminApi.listUsers.mockResolvedValue([user]);
    adminApi.updateUser.mockResolvedValue({ ...user, approval_state: nextState });
    render(<AdminPanel token="firebase-token" onSignOut={vi.fn()} />);

    await actor.click(await screen.findByRole("button", { name: label }));
    await waitFor(() => expect(adminApi.updateUser).toHaveBeenCalledWith("firebase-token", user.user_id, nextState));
  });

  it("shows the approved-user ceiling and aggregate-only metric cards", async () => {
    adminApi.listUsers.mockResolvedValue([approvedUser, { ...approvedUser, user_id: "user-3" }]);
    render(<AdminPanel token="firebase-token" onSignOut={vi.fn()} />);

    expect(await screen.findByText("2 / 10")).toBeInTheDocument();
    expect(screen.getByText("Requests today")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.queryByText(/question|answer|passage|trace/i)).not.toBeInTheDocument();
  });

  it("reports forbidden access distinctly", async () => {
    adminApi.listUsers.mockRejectedValue({ status: 403 });
    render(<AdminPanel token="firebase-token" onSignOut={vi.fn()} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/do not have administrator access/i);
  });
});
