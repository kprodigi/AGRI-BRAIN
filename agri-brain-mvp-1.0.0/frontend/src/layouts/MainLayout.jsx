import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip";
import {
  LayoutDashboard,
  Leaf,
  ShieldCheck,
  MapPin,
  BarChart3,
  Settings,
  Sun,
  Moon,
  Bell,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Wifi,
  WifiOff,
  User,
  Brain,
  Play,
  MessageCircle,
} from "lucide-react";

const NAV_ITEMS = [
  { key: "ops", label: "Operations", icon: LayoutDashboard, path: "/" },
  { key: "quality", label: "Quality", icon: ShieldCheck, path: "/quality" },
  { key: "decisions", label: "Decisions", icon: Leaf, path: "/decisions" },
  { key: "map", label: "Map", icon: MapPin, path: "/map" },
  { key: "analytics", label: "Analytics", icon: BarChart3, path: "/analytics" },
  { key: "mcp-pirag", label: "MCP/piRAG", icon: Brain, path: "/mcp-pirag" },
  { key: "demo", label: "Demo", icon: Play, path: "/demo" },
];

export default function MainLayout({ children, wsConnected, notifications, unreadCount, onMarkAllRead }) {
  const { theme, setTheme, isDark } = useTheme();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [bellOpen, setBellOpen] = useState(false);

  const currentPath = location.pathname;

  const toggleTheme = () => {
    if (isDark) setTheme("light");
    else setTheme("dark");
  };

  // Breadcrumb
  const currentItem = NAV_ITEMS.find((i) => i.path === currentPath) || NAV_ITEMS[0];

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex h-screen overflow-hidden bg-background">
        {/* Mobile overlay */}
        {mobileOpen && (
          <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setMobileOpen(false)} />
        )}

        {/* Sidebar */}
        <aside
          className={cn(
            "fixed lg:relative z-50 flex flex-col h-full bg-sidebar border-r border-sidebar-border transition-all duration-300",
            collapsed ? "w-16" : "w-60",
            mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
          )}
        >
          {/* Logo */}
          <div className={cn("flex items-center gap-3 px-4 h-14 border-b border-sidebar-border", collapsed && "justify-center px-2")}>
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary text-primary-foreground font-bold text-sm shrink-0">
              AB
            </div>
            {!collapsed && (
              <div className="flex flex-col min-w-0">
                <span className="font-semibold text-sm text-sidebar-foreground truncate">AGRI-BRAIN</span>
                <span className="text-[10px] text-muted-foreground">Supply Chain Gov.</span>
              </div>
            )}
          </div>

          {/* Nav items */}
          <nav className="flex-1 py-3 px-2 space-y-1 overflow-y-auto">
            {NAV_ITEMS.map((item) => {
              const active = currentPath === item.path;
              return (
                <Tooltip key={item.key}>
                  <TooltipTrigger asChild>
                    <Link
                      to={item.path}
                      onClick={() => setMobileOpen(false)}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                        active
                          ? "bg-sidebar-accent text-sidebar-primary"
                          : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                        collapsed && "justify-center px-2"
                      )}
                    >
                      <item.icon className="w-5 h-5 shrink-0" />
                      {!collapsed && <span>{item.label}</span>}
                    </Link>
                  </TooltipTrigger>
                  {collapsed && (
                    <TooltipContent side="right">{item.label}</TooltipContent>
                  )}
                </Tooltip>
              );
            })}
          </nav>

          {/* Bottom section */}
          <div className="border-t border-sidebar-border p-2 space-y-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  to="/admin"
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                    currentPath === "/admin"
                      ? "bg-sidebar-accent text-sidebar-primary"
                      : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    collapsed && "justify-center px-2"
                  )}
                >
                  <Settings className="w-5 h-5 shrink-0" />
                  {!collapsed && <span>Admin Panel</span>}
                </Link>
              </TooltipTrigger>
              {collapsed && <TooltipContent side="right">Admin Panel</TooltipContent>}
            </Tooltip>

            {/* Collapse toggle (desktop only) */}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="hidden lg:flex items-center justify-center w-full px-3 py-2 rounded-lg text-sidebar-foreground/50 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
            >
              {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
            </button>
          </div>
        </aside>

        {/* Main content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Top header */}
          <header className="sticky top-0 z-30 flex items-center justify-between h-14 px-4 lg:px-6 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="flex items-center gap-3">
              {/* Mobile menu toggle */}
              <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(!mobileOpen)}>
                {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
              {/* Breadcrumb */}
              <div className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground">AGRI-BRAIN</span>
                <span className="text-muted-foreground">/</span>
                <span className="font-medium">{currentItem.label}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* WS status */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium">
                    {wsConnected ? (
                      <>
                        <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                        </span>
                        <span className="hidden sm:inline text-emerald-600 dark:text-emerald-400">Live</span>
                      </>
                    ) : (
                      <>
                        <span className="h-2 w-2 rounded-full bg-red-500" />
                        <span className="hidden sm:inline text-red-600 dark:text-red-400">Offline</span>
                      </>
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent>{wsConnected ? "WebSocket connected" : "WebSocket disconnected"}</TooltipContent>
              </Tooltip>

              {/* Theme toggle */}
              <Button variant="ghost" size="icon" onClick={toggleTheme} className="h-8 w-8">
                {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </Button>

              {/* Notification bell */}
              <div className="relative">
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setBellOpen(!bellOpen)}>
                  <Bell className="w-4 h-4" />
                  {unreadCount > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[10px] font-bold text-destructive-foreground">
                      {unreadCount > 99 ? "99+" : unreadCount}
                    </span>
                  )}
                </Button>
                {/* Notification dropdown */}
                {bellOpen && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setBellOpen(false)} />
                    <div className="absolute right-0 top-10 z-50 w-80 rounded-lg border bg-popover text-popover-foreground shadow-lg">
                      <div className="flex items-center justify-between p-3 border-b">
                        <span className="font-semibold text-sm">Notifications</span>
                        {unreadCount > 0 && (
                          <button onClick={() => { onMarkAllRead?.(); }} className="text-xs text-primary hover:underline">
                            Mark all read
                          </button>
                        )}
                      </div>
                      <div className="max-h-64 overflow-y-auto">
                        {notifications.length === 0 ? (
                          <div className="p-4 text-center text-sm text-muted-foreground">No notifications</div>
                        ) : (
                          notifications.slice(0, 20).map((n) => (
                            <div key={n.id} className={cn("px-3 py-2 border-b last:border-0 text-sm", !n.read && "bg-accent/50")}>
                              <div className="flex items-center gap-2">
                                <span className={cn(
                                  "h-1.5 w-1.5 rounded-full shrink-0",
                                  n.type === "error" ? "bg-destructive" : n.type === "warning" ? "bg-warning" : "bg-primary"
                                )} />
                                <span className="font-medium truncate">{n.title}</span>
                              </div>
                              <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">{n.message}</p>
                              <p className="text-[10px] text-muted-foreground mt-0.5">
                                {new Date(n.timestamp).toLocaleTimeString()}
                              </p>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* User avatar placeholder */}
              <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                <User className="w-4 h-4 text-primary" />
              </div>
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 overflow-y-auto pb-16 lg:pb-0">
            <div className="p-4 lg:p-6 max-w-[1600px] mx-auto">
              {children}
            </div>
          </main>
        </div>

        {/* Mobile bottom navigation */}
        <nav className="fixed bottom-0 left-0 right-0 z-40 lg:hidden border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex items-center justify-around h-14">
            {NAV_ITEMS.map((item) => {
              const active = currentPath === item.path;
              return (
                <Link
                  key={item.key}
                  to={item.path}
                  className={cn(
                    "flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg text-xs transition-colors",
                    active ? "text-primary" : "text-muted-foreground"
                  )}
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </nav>
      </div>
    </TooltipProvider>
  );
}
