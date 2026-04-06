import React, { useEffect, useState, useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn, fmt, short, mcpRaw, mcpCall, mcpLog, getApiKey } from "@/lib/utils";
import { getApiBase } from "@/mvp/api.js";
import { toast } from "sonner";
import {
  Wrench, Database, FileText, Play, ScrollText, Search, RefreshCw,
  Loader2, ChevronDown, ChevronRight, BookOpen, Zap, Copy,
} from "lucide-react";

const API = getApiBase();

const TOOL_GROUPS = {
  regulatory: { label: "Domain", color: "bg-teal-500/10 text-teal-600 dark:text-teal-400" },
  temperature: { label: "Domain", color: "bg-teal-500/10 text-teal-600 dark:text-teal-400" },
  quality: { label: "Domain", color: "bg-teal-500/10 text-teal-600 dark:text-teal-400" },
  supply_chain: { label: "Domain", color: "bg-teal-500/10 text-teal-600 dark:text-teal-400" },
  environmental: { label: "Domain", color: "bg-teal-500/10 text-teal-600 dark:text-teal-400" },
  retrieval: { label: "Intelligence", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" },
  explainability: { label: "Intelligence", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" },
  context: { label: "Intelligence", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" },
  math: { label: "Utility", color: "bg-gray-500/10 text-gray-600 dark:text-gray-400" },
  conversion: { label: "Utility", color: "bg-gray-500/10 text-gray-600 dark:text-gray-400" },
  simulation: { label: "Utility", color: "bg-gray-500/10 text-gray-600 dark:text-gray-400" },
  governance: { label: "Utility", color: "bg-gray-500/10 text-gray-600 dark:text-gray-400" },
};

function getToolColor(capabilities) {
  for (const cap of (capabilities || [])) {
    const grp = TOOL_GROUPS[cap];
    if (grp) return grp;
  }
  return { label: "Other", color: "bg-gray-500/10 text-gray-600" };
}

// ===================== Tool Browser =====================
function ToolBrowser({ tools, onSelectTool }) {
  const grouped = { Domain: [], Intelligence: [], Utility: [] };
  for (const tool of tools) {
    const grp = getToolColor(tool.capabilities);
    const cat = grouped[grp.label] || (grouped["Other"] = []);
    cat.push({ ...tool, _grp: grp });
  }

  return (
    <div className="space-y-6">
      {Object.entries(grouped).map(([label, items]) => {
        if (!items || items.length === 0) return null;
        return (
          <div key={label}>
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-wider">{label} Tools</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {items.map((tool) => (
                <Card
                  key={tool.name}
                  className="cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => onSelectTool(tool)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-sm font-mono">{tool.name}</h4>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                          {tool.description}
                        </p>
                      </div>
                      <Wrench className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
                    </div>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {(tool.capabilities || []).slice(0, 3).map((cap) => {
                        const g = TOOL_GROUPS[cap];
                        return (
                          <Badge key={cap} className={cn("text-[9px] border-0", g?.color || "bg-muted")}>
                            {cap}
                          </Badge>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ===================== Resource Monitor =====================
function ResourceMonitor() {
  const [resources, setResources] = useState([]);
  const [values, setValues] = useState({});
  const [loading, setLoading] = useState(true);

  const loadResources = useCallback(async () => {
    try {
      const key = getApiKey();
      const headers = key ? { "x-api-key": key } : {};
      const res = await fetch(`${API}/mcp/resources`, { headers });
      const data = await res.json();
      setResources(data.resources || []);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  const readValues = useCallback(async () => {
    const vals = {};
    for (const r of resources) {
      try {
        const result = await mcpRaw(API, "resources/read", { uri: r.uri });
        const text = result?.contents?.[0]?.text;
        vals[r.uri] = text ? (text.length < 100 ? text : text.substring(0, 100) + "...") : "N/A";
      } catch {
        vals[r.uri] = "error";
      }
    }
    setValues(vals);
  }, [resources]);

  useEffect(() => { loadResources(); }, [loadResources]);

  useEffect(() => {
    if (resources.length === 0) return;
    readValues();
    const id = setInterval(readValues, 5000);
    return () => clearInterval(id);
  }, [resources, readValues]);

  if (loading) return <Skeleton className="h-48" />;

  return (
    <Card>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>URI</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Current Value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {resources.length === 0 ? (
              <TableRow><TableCell colSpan={3} className="text-center text-muted-foreground py-8">No resources found.</TableCell></TableRow>
            ) : (
              resources.map((r) => (
                <TableRow key={r.uri}>
                  <TableCell className="font-mono text-xs">{r.uri}</TableCell>
                  <TableCell className="text-sm">{r.name}</TableCell>
                  <TableCell className="font-mono text-xs max-w-xs truncate">
                    {values[r.uri] ?? <Loader2 className="w-3 h-3 animate-spin" />}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}

// ===================== Prompt Browser =====================
function PromptBrowser() {
  const [prompts, setPrompts] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [forms, setForms] = useState({});
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const key = getApiKey();
        const headers = key ? { "x-api-key": key } : {};
        const res = await fetch(`${API}/mcp/prompts`, { headers });
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        setPrompts(data.prompts || []);
      } catch { /* ignore */ }
      setLoading(false);
    })();
  }, []);

  const expand = async (name) => {
    try {
      const args = forms[name] || {};
      const result = await mcpRaw(API, "prompts/get", { name, arguments: args });
      setResults((r) => ({ ...r, [name]: result }));
    } catch (e) {
      toast.error(`Failed: ${e.message}`);
    }
  };

  if (loading) return <Skeleton className="h-48" />;

  return (
    <div className="space-y-3">
      {prompts.length === 0 && (
        <Card><CardContent className="p-8 text-center text-muted-foreground">No prompts found.</CardContent></Card>
      )}
      {prompts.map((p) => (
        <Card key={p.name}>
          <CardContent className="p-4">
            <button
              onClick={() => setExpanded(expanded === p.name ? null : p.name)}
              className="w-full flex items-center justify-between text-left"
            >
              <div>
                <h4 className="font-semibold text-sm font-mono">{p.name}</h4>
                {p.description && <p className="text-xs text-muted-foreground mt-0.5">{p.description}</p>}
              </div>
              {expanded === p.name ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
            {expanded === p.name && (
              <div className="mt-3 space-y-3 pt-3 border-t">
                {(p.arguments || []).map((arg) => (
                  <div key={arg.name} className="grid grid-cols-3 gap-2 items-center">
                    <Label className="text-xs">{arg.name}{arg.required && <span className="text-red-500">*</span>}</Label>
                    <Input
                      className="col-span-2 h-8 text-xs font-mono"
                      placeholder={arg.description || arg.name}
                      value={(forms[p.name] || {})[arg.name] || ""}
                      onChange={(e) => setForms((f) => ({
                        ...f, [p.name]: { ...(f[p.name] || {}), [arg.name]: e.target.value },
                      }))}
                    />
                  </div>
                ))}
                <Button size="sm" onClick={() => expand(p.name)}>
                  <Play className="w-3 h-3 mr-1" /> Expand Query
                </Button>
                {results[p.name] && (
                  <pre className="mt-2 p-3 rounded-md bg-muted text-xs font-mono overflow-x-auto max-h-48">
                    {JSON.stringify(results[p.name], null, 2)}
                  </pre>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ===================== Live Tool Invocation =====================
function LiveInvocation({ tools, selectedTool: initialTool }) {
  const [selected, setSelected] = useState(initialTool?.name || "");
  const [args, setArgs] = useState({});
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (initialTool) {
      setSelected(initialTool.name);
      setArgs({});
      setResult(null);
    }
  }, [initialTool]);

  const tool = tools.find((t) => t.name === selected);
  const schema = tool?.inputSchema?.properties || {};

  const PRESETS = {
    check_compliance: { temperature: "14.0", humidity: "85.0", product_type: "spinach" },
    pirag_query: { query: "FDA temperature violation corrective action", k: "4" },
    explain: { action: "local_redistribute", role: "farm", scenario: "heatwave", rho: "0.35", temperature: "14.0" },
  };

  const applyPreset = () => {
    const preset = PRESETS[selected];
    if (preset) setArgs(preset);
  };

  const run = async () => {
    setRunning(true);
    try {
      const typedArgs = {};
      for (const [k, v] of Object.entries(args)) {
        const prop = schema[k];
        if (prop?.type === "number" || prop?.type === "integer") {
          typedArgs[k] = Number(v);
        } else if (prop?.type === "boolean") {
          typedArgs[k] = v === "true";
        } else {
          typedArgs[k] = v;
        }
      }
      const res = await mcpCall(API, selected, typedArgs);
      setResult(res);
    } catch (e) {
      setResult({ error: e.message });
      toast.error(`Tool error: ${e.message}`);
    }
    setRunning(false);
  };

  const renderResult = () => {
    if (!result) return null;

    // Color-coded compliance status
    if (selected === "check_compliance" && result.compliant !== undefined) {
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge className={result.compliant ? "bg-emerald-500/10 text-emerald-600 border-0" : "bg-red-500/10 text-red-600 border-0"}>
              {result.compliant ? "Compliant" : "Violation"}
            </Badge>
            {result.severity && <Badge variant="outline" className="text-[10px]">{result.severity}</Badge>}
          </div>
          <pre className="p-3 rounded-md bg-muted text-xs font-mono overflow-x-auto max-h-64">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      );
    }

    // piRAG results with keyword tags
    if (selected === "pirag_query" && result.results) {
      return (
        <div className="space-y-2">
          {(result.results || []).map((doc, i) => (
            <Card key={i} className="bg-muted/30">
              <CardContent className="p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs font-semibold">{doc.doc_id}</span>
                  <Badge variant="outline" className="text-[10px]">score: {fmt(doc.score, 3)}</Badge>
                </div>
                <div className="mt-1 h-1.5 rounded-full bg-muted overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500" style={{ width: `${(doc.score || 0) * 100}%` }} />
                </div>
                {doc.keywords && doc.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {doc.keywords.map((kw, j) => (
                      <Badge key={j} className="text-[9px] bg-purple-500/10 text-purple-600 border-0">{kw}</Badge>
                    ))}
                  </div>
                )}
                {doc.passage && (
                  <p className="text-xs text-muted-foreground mt-2 line-clamp-3">{doc.passage}</p>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      );
    }

    // Explanation with BECAUSE/WITHOUT highlighting
    if (selected === "explain" && result.full_explanation) {
      const text = result.full_explanation;
      return (
        <div className="space-y-2">
          {result.causal_chain?.primary_cause && (
            <Badge className="bg-teal-500/10 text-teal-600 border-0">Primary: {result.causal_chain.primary_cause}</Badge>
          )}
          <div className="p-3 rounded-md bg-muted text-xs leading-relaxed">
            {text.split(/(BECAUSE|WITHOUT)/g).map((part, i) => {
              if (part === "BECAUSE") return <span key={i} className="font-bold text-teal-600">BECAUSE</span>;
              if (part === "WITHOUT") return <span key={i} className="font-bold text-amber-600">WITHOUT</span>;
              return <span key={i}>{part}</span>;
            })}
          </div>
        </div>
      );
    }

    return (
      <pre className="p-3 rounded-md bg-muted text-xs font-mono overflow-x-auto max-h-64">
        {JSON.stringify(result, null, 2)}
      </pre>
    );
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Zap className="w-4 h-4 text-primary" /> Live Tool Invocation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Select value={selected} onValueChange={(v) => { setSelected(v); setArgs({}); setResult(null); }}>
            <SelectTrigger className="flex-1"><SelectValue placeholder="Select a tool..." /></SelectTrigger>
            <SelectContent>
              {tools.map((t) => (
                <SelectItem key={t.name} value={t.name}>{t.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {PRESETS[selected] && (
            <Button variant="outline" size="sm" onClick={applyPreset}>Preset</Button>
          )}
        </div>

        {tool && Object.keys(schema).length > 0 && (
          <div className="space-y-2">
            {Object.entries(schema).map(([key, prop]) => (
              <div key={key} className="grid grid-cols-3 gap-2 items-center">
                <Label className="text-xs">
                  {key}
                  {tool.inputSchema?.required?.includes(key) && <span className="text-red-500">*</span>}
                </Label>
                <Input
                  className="col-span-2 h-8 text-xs font-mono"
                  placeholder={prop.description || `${prop.type || "string"}`}
                  value={args[key] || ""}
                  onChange={(e) => setArgs((a) => ({ ...a, [key]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        )}

        <Button onClick={run} disabled={!selected || running}>
          {running ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
          Run Tool
        </Button>

        {result && (
          <div className="mt-2">{renderResult()}</div>
        )}
      </CardContent>
    </Card>
  );
}

// ===================== piRAG Search =====================
function PiragSearch() {
  const [query, setQuery] = useState("");
  const [role, setRole] = useState("farm");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);

  const search = async () => {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const res = await mcpCall(API, "pirag_query", {
        query: query.trim(),
        k: 5,
        role,
        temperature: 14.0,
        rho: 0.3,
        physics_expansion: true,
        physics_reranking: true,
      });
      setResults(res);
    } catch (e) {
      toast.error(`Search failed: ${e.message}`);
    }
    setSearching(false);
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-48">
              <Label className="text-xs mb-1.5 block">Query</Label>
              <div className="relative">
                <Search className="absolute left-2.5 top-2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search the piRAG knowledge base..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && search()}
                  className="pl-8"
                />
              </div>
            </div>
            <div className="w-32">
              <Label className="text-xs mb-1.5 block">Role</Label>
              <Select value={role} onValueChange={setRole}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {["farm", "processor", "cooperative", "distributor", "recovery"].map((r) => (
                    <SelectItem key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={search} disabled={searching}>
              {searching ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Search className="w-4 h-4 mr-1" />}
              Search
            </Button>
          </div>
        </CardContent>
      </Card>

      {results && (
        <div className="space-y-2">
          {results.physics_expanded && (
            <Badge className="bg-blue-500/10 text-blue-600 border-0 text-[10px]">Physics-expanded query</Badge>
          )}
          {(results.results || []).map((doc, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-blue-500" />
                    <span className="font-mono text-sm font-semibold">{doc.doc_id}</span>
                  </div>
                  <span className="font-mono text-xs text-muted-foreground">
                    score: {fmt(doc.score, 3)}
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden mb-2">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all"
                    style={{ width: `${Math.min((doc.score || 0) * 100, 100)}%` }}
                  />
                </div>
                {doc.keywords && doc.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {doc.keywords.map((kw, j) => (
                      <Badge key={j} className="text-[9px] bg-purple-500/10 text-purple-600 dark:text-purple-400 border-0">{kw}</Badge>
                    ))}
                  </div>
                )}
                {doc.passage && (
                  <p className="text-xs text-muted-foreground line-clamp-4">{doc.passage}</p>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

// ===================== Protocol Log =====================
function ProtocolLog() {
  const [log, setLog] = useState([...mcpLog]);
  const intervalRef = useRef(null);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setLog([...mcpLog]);
    }, 1000);
    return () => clearInterval(intervalRef.current);
  }, []);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <ScrollText className="w-4 h-4" /> Protocol Interaction Log
          </CardTitle>
          <Badge variant="outline" className="text-[10px]">{log.length} entries</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-72">
          {log.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No MCP interactions yet. Use the tools above to generate protocol traffic.
            </p>
          ) : (
            <div className="space-y-2">
              {[...log].reverse().map((entry, i) => (
                <div key={i} className="flex items-start gap-2 text-xs p-2 rounded-md bg-muted/50">
                  <Badge
                    className={cn(
                      "text-[9px] shrink-0 border-0 mt-0.5",
                      entry.status === "success" ? "bg-emerald-500/10 text-emerald-600" : "bg-red-500/10 text-red-600"
                    )}
                  >
                    {entry.status}
                  </Badge>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-semibold">{entry.method}</span>
                      <span className="text-muted-foreground">{entry.ts?.split("T")[1]?.split(".")[0]}</span>
                    </div>
                    <p className="font-mono text-muted-foreground truncate mt-0.5">
                      {entry.preview}
                    </p>
                  </div>
                  <button
                    onClick={() => { navigator.clipboard.writeText(JSON.stringify(entry, null, 2)); toast.success("Copied"); }}
                    className="text-muted-foreground hover:text-primary shrink-0"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// ===================== Main MCP Tab =====================
export default function McpTab() {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTool, setSelectedTool] = useState(null);
  const [activeSubTab, setActiveSubTab] = useState("tools");

  useEffect(() => {
    (async () => {
      try {
        const result = await mcpRaw(API, "tools/list");
        setTools(result?.tools || []);
      } catch (e) {
        toast.error(`Failed to load MCP tools: ${e.message}`);
      }
      setLoading(false);
    })();
  }, []);

  const handleSelectTool = (tool) => {
    setSelectedTool(tool);
    setActiveSubTab("invoke");
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => <Skeleton key={i} className="h-24" />)}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Wrench className="w-5 h-5 text-primary" /> MCP Explorer
          </h2>
          <p className="text-sm text-muted-foreground">
            Model Context Protocol interoperability layer &mdash; {tools.length} tools, JSON-RPC 2.0
          </p>
        </div>
        <Badge variant="outline" className="font-mono text-xs">{tools.length} tools</Badge>
      </div>

      <Tabs value={activeSubTab} onValueChange={setActiveSubTab}>
        <TabsList>
          <TabsTrigger value="tools" className="text-xs"><Wrench className="w-3 h-3 mr-1" /> Tools</TabsTrigger>
          <TabsTrigger value="resources" className="text-xs"><Database className="w-3 h-3 mr-1" /> Resources</TabsTrigger>
          <TabsTrigger value="prompts" className="text-xs"><FileText className="w-3 h-3 mr-1" /> Prompts</TabsTrigger>
          <TabsTrigger value="invoke" className="text-xs"><Zap className="w-3 h-3 mr-1" /> Invoke</TabsTrigger>
          <TabsTrigger value="pirag" className="text-xs"><BookOpen className="w-3 h-3 mr-1" /> piRAG Search</TabsTrigger>
          <TabsTrigger value="log" className="text-xs"><ScrollText className="w-3 h-3 mr-1" /> Protocol Log</TabsTrigger>
        </TabsList>

        <div className="mt-4">
          <TabsContent value="tools">
            <ToolBrowser tools={tools} onSelectTool={handleSelectTool} />
          </TabsContent>
          <TabsContent value="resources">
            <ResourceMonitor />
          </TabsContent>
          <TabsContent value="prompts">
            <PromptBrowser />
          </TabsContent>
          <TabsContent value="invoke">
            <LiveInvocation tools={tools} selectedTool={selectedTool} />
          </TabsContent>
          <TabsContent value="pirag">
            <PiragSearch />
          </TabsContent>
          <TabsContent value="log">
            <ProtocolLog />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
