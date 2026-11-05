export const number = (value: unknown, digits = 3) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-GB", { maximumFractionDigits: digits })
    : "Unavailable";
export const percent = (value: unknown, digits = 2) =>
  typeof value === "number" && Number.isFinite(value)
    ? `${value > 0 ? "+" : ""}${number(value, digits)}%`
    : "Unavailable";
export function dateLabel(value?: string | null) {
  if (!value) return "Date unavailable";
  // Market observations are calendar dates, not instants to shift into UTC.
  const date = new Date(
    /^\d{4}-\d{2}-\d{2}/.test(value) ? value.slice(0, 10) : value,
  );
  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : date.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      });
}
export function safeLink(value?: string) {
  try {
    const url = new URL(value || "");
    return ["http:", "https:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}
