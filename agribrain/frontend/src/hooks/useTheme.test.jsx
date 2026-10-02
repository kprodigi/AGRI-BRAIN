import { describe, expect, it, beforeEach, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { ThemeProvider, useTheme } from "./useTheme";

function ThemeProbe({ onReady }) {
  const ctx = useTheme();
  onReady(ctx);
  return <span data-testid="theme">{ctx.theme}</span>;
}

describe("useTheme", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = "";
    vi.stubGlobal("matchMedia", (q) => ({
      matches: q.includes("dark"),
      media: q,
      addEventListener: () => {},
      removeEventListener: () => {},
    }));
  });

  it("uses the provided defaultTheme when nothing is stored", () => {
    let captured;
    render(
      <ThemeProvider defaultTheme="light">
        <ThemeProbe onReady={(c) => (captured = c)} />
      </ThemeProvider>
    );
    expect(captured.theme).toBe("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
  });

  it("persists the theme to localStorage when setTheme is called", () => {
    let captured;
    render(
      <ThemeProvider defaultTheme="light" storageKey="test-theme">
        <ThemeProbe onReady={(c) => (captured = c)} />
      </ThemeProvider>
    );
    act(() => captured.setTheme("dark"));
    expect(localStorage.getItem("test-theme")).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("reads an existing stored theme on mount", () => {
    localStorage.setItem("agri-brain-theme", "dark");
    let captured;
    render(
      <ThemeProvider>
        <ThemeProbe onReady={(c) => (captured = c)} />
      </ThemeProvider>
    );
    expect(captured.theme).toBe("dark");
  });
});
