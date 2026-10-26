import { expect, test } from "@playwright/test";

test.describe("v2.1 admin evaluation journey", () => {
  test.skip(!process.env.V21_E2E_BASE_URL, "Set V21_E2E_BASE_URL for the final-image environment.");

  test("keeps the admin journey accessible across desktop and mobile", async ({ page }) => {
    await page.goto(process.env.V21_E2E_BASE_URL!);
    await expect(page.getByRole("heading", { name: /sign in|log in/i })).toBeVisible();
    await expect(page.getByRole("main")).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByRole("main")).toBeVisible();
    await expect(page.locator("body")).toHaveScreenshot("v21-admin-mobile.png", { animations: "disabled" });
  });

  test("does not expose admin evaluation controls to an unauthenticated user", async ({ page }) => {
    await page.goto(process.env.V21_E2E_BASE_URL!);
    await expect(page.getByRole("heading", { name: /evaluation and release decision/i })).toHaveCount(0);
    await expect(page.getByText(/administrator access/i)).toHaveCount(0);
  });
});
