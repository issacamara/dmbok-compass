import { beforeEach, describe, expect, it, vi } from "vitest";
import { getConfiguration, listAggregateMetrics, listUsers, updateConfiguration, updateUser } from "./admin";

describe("administrator API", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("sends the Firebase token and approval filter when listing users", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("[]", { status: 200 }));
    await listUsers("firebase-token", "pending");
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/users?approval_state=pending", expect.objectContaining({
      headers: { Authorization: "Bearer firebase-token" },
    }));
  });

  it("updates a user and quota policy through the admin routes", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) =>
      new Response(JSON.stringify({}), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    await updateUser("token", "user/1", "approved");
    await getConfiguration("token");
    await updateConfiguration("token", { per_user_daily_limit: 20, global_daily_limit: 50 });

    expect(fetchMock.mock.calls.map(([input]) => input)).toEqual([
      "/api/admin/users/user%2F1", "/api/admin/configuration", "/api/admin/configuration",
    ]);
    expect(fetchMock.mock.calls[0][1]).toEqual(expect.objectContaining({
      method: "PATCH", body: JSON.stringify({ approval_state: "approved" }),
    }));
    expect(fetchMock.mock.calls[2][1]).toEqual(expect.objectContaining({
      method: "PATCH", body: JSON.stringify({ per_user_daily_limit: 20, global_daily_limit: 50 }),
    }));
  });

  it("reads content-free aggregate metrics through the guarded admin route", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("[]", { status: 200 }));
    await listAggregateMetrics("firebase-token");
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/metrics", expect.objectContaining({
      headers: { Authorization: "Bearer firebase-token" },
    }));
  });
});
