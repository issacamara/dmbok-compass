import { beforeEach, describe, expect, it, vi } from "vitest";
import { approveCorpusStagingAttempt, getConfiguration, getCorpusStagingStatus, getCorpusStagingTemplate, listAggregateMetrics, listUsers, submitCorpusStagingAttempt, updateConfiguration, updateUser, validateCorpusStagingAttempt } from "./admin";

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

  it("uses the admin staging endpoints with multipart uploads and the reviewed revision", async () => {
    const response = { attempt_id: "attempt/1", revision: 7, state: "awaiting_review" };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(response), { status: 200 }));
    const pdf = new File(["pdf"], "source.pdf", { type: "application/pdf" });
    const json = new File(["{}"], "structure.json", { type: "application/json" });
    await getCorpusStagingTemplate("token");
    await submitCorpusStagingAttempt("token", pdf, json);
    await validateCorpusStagingAttempt("token", "attempt/1");
    await approveCorpusStagingAttempt("token", "attempt/1", 7);
    await getCorpusStagingStatus("token", "attempt/1");

    expect(fetchMock.mock.calls.map(([input]) => input)).toEqual([
      "/api/admin/corpus-staging/template", "/api/admin/corpus-staging/attempts",
      "/api/admin/corpus-staging/attempts/attempt%2F1/validate",
      "/api/admin/corpus-staging/attempts/attempt%2F1/approve?expected_revision=7",
      "/api/admin/corpus-staging/attempts/attempt%2F1",
    ]);
    const upload = fetchMock.mock.calls[1][1];
    expect(upload?.method).toBe("POST");
    expect(upload?.headers).toEqual({ Authorization: "Bearer token" });
    expect(upload?.body).toBeInstanceOf(FormData);
    expect(fetchMock.mock.calls[3][1]).toEqual(expect.objectContaining({ method: "POST" }));
  });
});
