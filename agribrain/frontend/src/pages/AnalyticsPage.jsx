import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
  CartesianGrid, Legend, ErrorBar, ReferenceLine,
} from "recharts";
import { BarChart3, TrendingUp, Route, FlaskConical } from "lucide-react";
import results from "@/data/studyResults.json";

const MODE_LABELS = { static: "Static", no_context: "No-Context", agribrain: "AGRI-BRAIN" };
const MODE_COLORS = { static: "#808080", no_context: "#4CAF50", agribrain: "#009688" };
const SCENARIO_LABELS = {
  heatwave: "Heatwave", overproduction: "Overproduction", cyber_outage: "Cyber outage",
  adaptive_pricing: "Adaptive pricing", baseline: "Baseline",
};
const ROUTE_LABELS = ["Cold chain", "Local redistribution", "Recovery"];
const METRIC_FORMAT = {
  ari: (v) => v.toFixed(4),
  waste: (v) => v.toFixed(3),
  carbon: (v) => v.toFixed(2),
  slca: (v) => v.toFixed(4),
  rle: (v) => v.toFixed(4),
  mean_decision_latency_ms: (v) => v.toFixed(3),
};

const signed = (v, digits = 4) => (v >= 0 ? "+" : "") + v.toFixed(digits);
const interval = (est, digits = 4) => `[${est.low.toFixed(digits)}, ${est.high.toFixed(digits)}]`;
const tooltipNumber = (v) => (typeof v === "number" ? v.toFixed(4) : v);

function StatCard({ label, value, sub }) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-2xl font-bold">{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </CardContent>
    </Card>
  );
}

function PrimaryTab() {
  const ariData = useMemo(() => results.meta.scenarios.map((s) => {
    const row = { scenario: SCENARIO_LABELS[s] };
    for (const mode of results.meta.modes) {
      const est = results.scenarios[s].metrics.ari[mode];
      row[mode] = est.mean;
      row[`${mode}_err`] = [est.mean - est.low, est.high - est.mean];
    }
    return row;
  }), []);
  const gainData = useMemo(() => results.meta.scenarios.map((s) => {
    const g = results.scenarios[s].paired_gain;
    return { scenario: SCENARIO_LABELS[s], gain: g.mean, gain_err: [g.mean - g.low, g.high - g.mean] };
  }), []);

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Adaptive Resilience Index by scenario</CardTitle>
            <CardDescription>Mean over 20 seeds with 95% BCa intervals</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ariData} barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0.3, 0.75]} tick={{ fontSize: 11 }} />
                  <ReTooltip contentStyle={{ fontSize: 12 }} formatter={tooltipNumber} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {results.meta.modes.map((mode) => (
                    <Bar key={mode} dataKey={mode} name={MODE_LABELS[mode]} fill={MODE_COLORS[mode]} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                      <ErrorBar dataKey={`${mode}_err`} width={6} strokeWidth={2} stroke="#1f2937" />
                    </Bar>
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Paired gain over No-Context</CardTitle>
            <CardDescription>AGRI-BRAIN minus No-Context ARI, 95% BCa interval; the 0.01 line is descriptive</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={gainData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <ReTooltip contentStyle={{ fontSize: 12 }} formatter={tooltipNumber} />
                  <ReferenceLine y={0.01} stroke="#6b7280" strokeDasharray="4 4" />
                  <Bar dataKey="gain" name="Paired gain" fill={MODE_COLORS.agribrain} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="gain_err" width={6} strokeWidth={2} stroke="#1f2937" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Overall outcomes</CardTitle>
          <CardDescription>Equal weight on the five scenarios within each seed; brackets are 95% BCa intervals across seeds</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                {results.meta.modes.map((m) => <TableHead key={m}>{MODE_LABELS[m]}</TableHead>)}
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(results.overall).map(([metric, entry]) => (
                <TableRow key={metric}>
                  <TableCell className="font-medium">
                    {entry.label}
                  </TableCell>
                  {results.meta.modes.map((m) => (
                    <TableCell key={m} className="tabular-nums">
                      {METRIC_FORMAT[metric](entry[m].mean)}
                      <span className="text-xs text-muted-foreground"> [{METRIC_FORMAT[metric](entry[m].low)}, {METRIC_FORMAT[metric](entry[m].high)}]</span>
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Paired gain by scenario</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Scenario</TableHead>
                <TableHead>Gain [95% BCa]</TableHead>
                <TableHead>Relative gain</TableHead>
                <TableHead>Different executed actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.meta.scenarios.map((s) => {
                const sc = results.scenarios[s];
                return (
                  <TableRow key={s}>
                    <TableCell className="font-medium">{SCENARIO_LABELS[s]}</TableCell>
                    <TableCell className="tabular-nums">{signed(sc.paired_gain.mean)} {interval(sc.paired_gain)}</TableCell>
                    <TableCell className="tabular-nums">{signed(sc.relative_gain_percent, 2)}%</TableCell>
                    <TableCell className="tabular-nums">{sc.different_actions_percent.mean.toFixed(1)}% {interval(sc.different_actions_percent, 1)}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function RoutingTab() {
  const data = ROUTE_LABELS.map((label, i) => ({
    route: label,
    no_context: results.routes_percent.no_context[i],
    agribrain: results.routes_percent.agribrain[i],
  }));
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Executed routes</CardTitle>
        <CardDescription>Share of executed actions, averaged over two-hour bins, five scenarios and 20 seeds</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="route" tick={{ fontSize: 11 }} />
              <YAxis unit="%" tick={{ fontSize: 11 }} />
              <ReTooltip contentStyle={{ fontSize: 12 }} formatter={(v) => `${v}%`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="no_context" name={MODE_LABELS.no_context} fill={MODE_COLORS.no_context} radius={[2, 2, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="agribrain" name={MODE_LABELS.agribrain} fill={MODE_COLORS.agribrain} radius={[2, 2, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          The context-enabled policy executes local redistribution more often. The two policies are adapted separately, so this
          describes executed routes, not the effect of a single channel.
        </p>
      </CardContent>
    </Card>
  );
}

const CONTROL_ROWS = [
  ["zero", "No adjustment", "ARI lost when the context adjustment is removed"],
  ["fixed", "Fixed mean adjustment", "ARI lost against the live adjustment"],
  ["shuffled", "Shuffled adjustment", "ARI lost against the live adjustment"],
];

function ControlsTab() {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Frozen-policy controls</CardTitle>
        <CardDescription>
          The trained AGRI-BRAIN policy re-evaluated with its context adjustment replaced; 100 policies per control. Mean ARI
          difference with 95% BCa interval. Exploratory; these are not separately adapted modes.
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Control</TableHead>
              <TableHead>Mean ARI difference [95% BCa]</TableHead>
              <TableHead>Reading</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {CONTROL_ROWS.map(([key, label, note]) => {
              const est = results.frozen_controls[key];
              return (
                <TableRow key={key}>
                  <TableCell className="font-medium">{label}</TableCell>
                  <TableCell className="tabular-nums">{est.mean.toFixed(4)} {interval(est)}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{note}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function SensitivityTab() {
  const data = results.sensitivity.map((s) => ({
    setting: s.setting.replaceAll("_", " "),
    gain: s.gain,
    gain_err: [s.gain - s.low, s.high - s.gain],
  }));
  const gains = results.sensitivity.map((s) => s.gain);
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Weight sensitivity</CardTitle>
        <CardDescription>
          Overall paired ARI gain over No-Context for {results.sensitivity.length} weight settings
          ({Math.min(...gains).toFixed(4)} to {Math.max(...gains).toFixed(4)}). Pointwise 95% BCa intervals, not simultaneous.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="setting" tick={{ fontSize: 9 }} interval={0} angle={-60} textAnchor="end" height={90} />
              <YAxis domain={[0, 0.03]} tick={{ fontSize: 11 }} />
              <ReTooltip contentStyle={{ fontSize: 12 }} formatter={tooltipNumber} />
              <ReferenceLine y={0.01} stroke="#6b7280" strokeDasharray="4 4" />
              <Bar dataKey="gain" name="Paired gain" fill={MODE_COLORS.agribrain} isAnimationActive={false}>
                <ErrorBar dataKey="gain_err" width={4} strokeWidth={1.5} stroke="#1f2937" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          The weights change the size of the gain, not its sign, over the tested ranges. The ranges are author-selected stress
          envelopes, not calibration intervals.
        </p>
      </CardContent>
    </Card>
  );
}

export default function AnalyticsPage() {
  const [tab, setTab] = useState("primary");
  const ari = results.overall.ari;
  const nominal = results.sensitivity.find((s) => s.setting === "nominal");
  return (
    <div className="space-y-6 pb-12">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-1">
          <BarChart3 className="w-6 h-6 text-primary" />
          <h1 className="text-2xl font-bold">Study results</h1>
          <Badge className="bg-teal-500/10 text-teal-600 border-0 text-xs">Static / No-Context / AGRI-BRAIN</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Synthetic spinach cold-chain study: {results.meta.seeds} seeds, five scenarios, {results.meta.primary_evaluations} evaluation
          episodes. All outcomes are modeled under the declared scoring assumptions, not field measurements.
        </p>
      </motion.div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Static ARI" value={ari.static.mean.toFixed(4)} />
        <StatCard label="No-Context ARI" value={ari.no_context.mean.toFixed(4)} />
        <StatCard label="AGRI-BRAIN ARI" value={ari.agribrain.mean.toFixed(4)} />
        <StatCard label="Paired gain over No-Context" value={signed(nominal.gain)} sub={`95% BCa ${interval(nominal)}`} />
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="w-full justify-start flex-wrap">
          <TabsTrigger value="primary" className="flex items-center gap-1.5"><BarChart3 className="w-3.5 h-3.5" /> Primary comparison</TabsTrigger>
          <TabsTrigger value="routing" className="flex items-center gap-1.5"><Route className="w-3.5 h-3.5" /> Routing</TabsTrigger>
          <TabsTrigger value="controls" className="flex items-center gap-1.5"><FlaskConical className="w-3.5 h-3.5" /> Frozen controls</TabsTrigger>
          <TabsTrigger value="sensitivity" className="flex items-center gap-1.5"><TrendingUp className="w-3.5 h-3.5" /> Weight sensitivity</TabsTrigger>
        </TabsList>
        <div className="mt-6">
          <TabsContent value="primary"><PrimaryTab /></TabsContent>
          <TabsContent value="routing"><RoutingTab /></TabsContent>
          <TabsContent value="controls"><ControlsTab /></TabsContent>
          <TabsContent value="sensitivity"><SensitivityTab /></TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
