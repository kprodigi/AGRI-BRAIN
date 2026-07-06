import React, { createContext, useContext, useEffect, useState } from "react";

const ThemeContext = createContext({ theme: "system", setTheme: () => {} });

export function ThemeProvider({ children, defaultTheme = "system", storageKey = "agri-brain-theme" }) {
  const [theme, setTheme] = useState(() => localStorage.getItem(storageKey) || defaultTheme);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");

    if (theme === "system") {
      const sys = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      root.classList.add(sys);
    } else {
      root.classList.add(theme);
    }
  }, [theme]);

  const value = {
    theme,
    setTheme: (t) => {
      localStorage.setItem(storageKey, t);
      setTheme(t);
    },
    isDark: theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches),
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);
