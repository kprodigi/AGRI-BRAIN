import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Button } from "./button";

describe("Button", () => {
  it("renders children and default classes", () => {
    render(<Button>Click me</Button>);
    const btn = screen.getByRole("button", { name: /click me/i });
    expect(btn).toBeInTheDocument();
    expect(btn.className).toMatch(/bg-primary/);
    expect(btn.tagName).toBe("BUTTON");
  });

  it("applies the destructive variant class", () => {
    render(<Button variant="destructive">Delete</Button>);
    const btn = screen.getByRole("button", { name: /delete/i });
    expect(btn.className).toMatch(/bg-destructive/);
  });

  it("forwards arbitrary props like disabled and onClick", () => {
    const onClick = () => {};
    render(
      <Button disabled onClick={onClick} data-testid="submit">
        Submit
      </Button>
    );
    const btn = screen.getByTestId("submit");
    expect(btn).toBeDisabled();
  });

  it("renders an anchor when asChild is set", () => {
    render(
      <Button asChild>
        <a href="/somewhere">go</a>
      </Button>
    );
    const link = screen.getByRole("link", { name: /go/i });
    expect(link).toBeInTheDocument();
    expect(link.getAttribute("href")).toBe("/somewhere");
  });
});
