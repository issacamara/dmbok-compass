import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./main";

describe("application shell", () => {
  it("renders the workspace identity and readiness state", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /grounded data management guidance/i })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("API workspace ready");
  });
});

