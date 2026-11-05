import type { Data, Layout } from "plotly.js";
export type MarketRow = {
  symbol: string;
  close: number | null;
  change_pct: number | null;
  volume: number | null;
  as_of: string | null;
};
export type Figure = { data: Data[]; layout: Partial<Layout> };
export type Chart = { title?: string; description?: string; figure: Figure };
export type Source = { title?: string; link?: string; date?: string };
export type Activity = { id: string; label: string; status: string };
export type StreamEvent = {
  type: string;
  id?: string;
  label?: string;
  status?: string;
  message?: string;
  text?: string;
  charts?: Chart[];
  sources?: Source[];
};
export type Message = {
  role: "user" | "assistant";
  content: string;
  charts?: Chart[];
  sources?: Source[];
  activity?: Activity[];
  elapsed?: number;
  error?: boolean;
};
export type Chat = { id: string; title: string; messages: Message[] };
export type Bar = {
  date: string;
  cloture: number;
  ouverture?: number;
  haut?: number;
  bas?: number;
  volume?: number;
};
export type Forecast = {
  implied_price?: number;
  predicted_log_return?: number;
  as_of?: string;
  forecast_for?: string;
  forecast_horizon_observations?: number;
  price_basis?: string;
  horizon_basis?: string;
  model_version?: string;
  [key: string]: unknown;
};
export type Stock = {
  columns: string[];
  data: Bar[];
  quality?: { invalid_ohlc_rows?: number; missing_volume_rows?: number };
  forecast?: Forecast;
  forecast_error?: string;
};
