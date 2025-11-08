

// Formats an ISO date/time string to Australia/Melbourne
export const fmtMelTime = (iso, { date = "medium", time = "short" } = {}) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-AU", {
    timeZone: "Australia/Melbourne",
    dateStyle: date,   // "full" | "long" | "medium" | "short"
    timeStyle: time,   // "full" | "long" | "medium" | "short"
  });
};

// Useful for axis ticks where you want just date or just time
export const fmtMelDateOnly = (iso) =>
  fmtMelTime(iso, { date: "medium", time: undefined });

export const fmtMelHour = (iso) =>
  new Date(iso).toLocaleTimeString("en-AU", {
    timeZone: "Australia/Melbourne",
    hour: "2-digit",
    minute: "2-digit",
  });
