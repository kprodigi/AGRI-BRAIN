import React from "react";
import { render, screen, within } from "@testing-library/react";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import AnalyticsPage from "./AnalyticsPage.jsx";
import { PRIMARY_PUBLICATION_MODES, SECONDARY_PUBLICATION_MODES } from "@/lib/publicationEvidence.js";

const scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"];
const modes = [...PRIMARY_PUBLICATION_MODES, ...SECONDARY_PUBLICATION_MODES];
const development_summary = Object.fromEntries(scenarios.map((scenario) => [scenario,
  Object.fromEntries(modes.map((mode) => [mode, {
    ari: 0.123, rle: 0.234, waste: 0.345, slca: 0.456, carbon: 12.345,
    equity: 0.567, decision_latency_ms: 7.891, constraint_violation_rate: 0.012,
  }])),
]));

beforeEach(() => {
  vi.spyOn(console, "warn").mockImplementation(() => {});
});
afterEach(() => vi.restoreAllMocks());

it("renders all development endpoints without publication claims or inference", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => ({
    ok: String(url).endsWith("/results/summary"), status: 503,
    json: async () => ({ ok: true, evidence_status: "development_only", publication_evidence: false, development_summary }),
  }));
  render(<AnalyticsPage />);
  expect(await screen.findByRole("status")).toHaveTextContent("One seed; not the certified 20-seed panel; not publication evidence");
  const tables = screen.getAllByRole("table");
  expect(within(tables[0]).getAllByRole("row")).toHaveLength(56);
  expect(within(tables[1]).getAllByRole("row")).toHaveLength(41);
  expect(within(tables[0]).getAllByText("7.891")).toHaveLength(55);
  expect(within(tables[0]).getAllByText("0.012")).toHaveLength(55);
  for (const mode of SECONDARY_PUBLICATION_MODES) {
    expect(within(tables[0]).getAllByText(mode)).toHaveLength(5);
  }
  expect(screen.queryByText(/Confirmatory Directional Evidence/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Backend-accepted artifact set/)).not.toBeInTheDocument();
  expect(screen.queryByText("Scenario Deep-Dive Gallery")).not.toBeInTheDocument();
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
});

it("retains the empty state when development results are unavailable", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: false, status: 503 });
  render(<AnalyticsPage />);
  expect(await screen.findByText("Publication evidence unavailable")).toBeInTheDocument();
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});

it("keeps validated publication data ahead of development results", async () => {
  const csv = (name, count) => `Scenario,${name},ARI,RLE,Waste,SLCA,Carbon,Equity\n`
    + Array.from({ length: count }, (_, i) => `${scenarios[i % 5]},${i % 2 ? "static" : "agribrain"},0.6,0.7,0.2,0.8,10,0.9`).join("\n");
  const summary = Object.fromEntries(scenarios.map((scenario) => [scenario,
    Object.fromEntries(["static", "agribrain"].map((mode) => [mode,
      Object.fromEntries(["ari", "rle", "waste", "carbon"].map((metric) => [metric, { mean: 0.5 }])),
    ])),
  ]));
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => ({
    ok: true,
    text: async () => String(url).includes("table1") ? csv("Method", 40) : csv("Variant", 30),
    json: async () => String(url).includes("benchmark_summary")
      ? { summary, _meta: { n_seeds: 20 } } : { significance: {} },
  }));
  render(<AnalyticsPage />);
  expect(await screen.findByText(/Backend-accepted artifact set/)).toBeInTheDocument();
  expect(screen.queryByText("Development results only")).not.toBeInTheDocument();
  expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/results/summary"))).toBe(false);
});
