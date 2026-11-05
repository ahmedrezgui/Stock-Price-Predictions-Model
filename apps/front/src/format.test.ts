import { expect, test } from "vitest";
import { dateLabel } from "./format";

test("observation labels preserve the source calendar day across timezone offsets", () => {
  expect(dateLabel("2026-10-12T00:00:00+01:00")).toBe("12 Oct 2026");
  expect(dateLabel("2026-10-12")).toBe("12 Oct 2026");
  expect(dateLabel(null)).toBe("Date unavailable");
});
