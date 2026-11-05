import {
  CheckCircle2,
  CircleDashed,
  AlertCircle,
  MinusCircle,
} from "lucide-react";
import type { Activity } from "../types";
export default function ExecutionStatus({
  activity,
}: {
  activity: Activity[];
}) {
  if (!activity.length) return null;
  return (
    <details className="execution-details">
      <summary>
        <span>Research activity</span>
        <span className="execution-count">
          {activity.filter((s) => s.status === "completed").length}/
          {activity.length}
        </span>
      </summary>
      <ol>
        {activity.map((step) => {
          const Icon =
            step.status === "completed"
              ? CheckCircle2
              : step.status === "running"
                ? CircleDashed
                : step.status === "failed"
                  ? AlertCircle
                  : MinusCircle;
          return (
            <li key={step.id} className={`execution-step ${step.status}`}>
              <Icon size={15} aria-hidden="true" />
              <span>
                {step.label} · {step.status}
              </span>
            </li>
          );
        })}
      </ol>
    </details>
  );
}
