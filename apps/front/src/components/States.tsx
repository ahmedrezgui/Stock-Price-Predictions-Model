import { AlertCircle, ArrowRight, Inbox } from "lucide-react";
export function LoadingState({
  label = "Loading workspace",
  chart = false,
}: {
  label?: string;
  chart?: boolean;
}) {
  return (
    <div
      role="status"
      className={chart ? "loading-state chart-skeleton" : "loading-state"}
    >
      <span className="sr-only">{label}</span>
      <div aria-hidden="true" className="skeleton-lines">
        <i />
        <i />
        <i />
        {chart && <div className="skeleton-chart" />}
      </div>
      <p className="text-xs text-muted mt-4">{label}…</p>
    </div>
  );
}
export function ErrorState({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div role="alert" className="error-state">
      <AlertCircle size={19} />
      <div>
        <strong>We couldn’t load this view</strong>
        <p>{message}</p>
        {retry && (
          <button className="text-link mt-3" onClick={retry}>
            Try again <ArrowRight size={14} />
          </button>
        )}
      </div>
    </div>
  );
}
export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: { label: string; run: () => void };
}) {
  return (
    <div className="empty-state">
      <Inbox size={25} />
      <h3>{title}</h3>
      {detail && <p>{detail}</p>}
      {action && (
        <button className="secondary mt-4" onClick={action.run}>
          {action.label}
        </button>
      )}
    </div>
  );
}
