import { test, expect, type Page } from "@playwright/test";
test.use({ colorScheme: "dark" });
async function settleCharts(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  await expect
    .poll(() =>
      page.locator(".plot-frame").evaluateAll((nodes) =>
        nodes
          .filter((n) => (n as HTMLElement).offsetWidth > 0)
          .every((n) => {
            const svg = n.querySelector(".main-svg");
            return (
              n.getAttribute("aria-busy") === "false" &&
              !!svg &&
              Math.abs(
                svg.getBoundingClientRect().width -
                  n.getBoundingClientRect().width,
              ) < 2
            );
          }),
      ),
    )
    .toBe(true);
}

test("overview recovers from failure and opens a dated one-step forecast", async ({
  page,
}) => {
  let available = false;
  const rows = Array.from({ length: 12 }, (_, i) => ({
    date: `2026-10-${String(i + 1).padStart(2, "0")}`,
    cloture: 100 + i / 4 + Math.sin(i),
    volume: 1000 + i * 100,
  }));
  const figure = {
    data: [
      {
        type: "scatter",
        mode: "lines",
        x: rows.map((r) => r.date),
        y: rows.map((r) => r.cloture),
        name: "BIAT",
      },
    ],
    layout: { xaxis: { type: "category", tickvals: rows.map((r) => r.date) } },
  };
  await page.route("**/market-api/market", (route) =>
    available
      ? route.fulfill({
          json: [
            {
              symbol: "BIAT",
              close: rows.at(-1)!.cloture,
              change_pct: 0.4,
              volume: 2100,
              as_of: "2026-10-12",
            },
          ],
        })
      : route.fulfill({
          status: 503,
          json: { detail: "Market snapshot unavailable" },
        }),
  );
  await page.route("**/market-api/companies", (route) =>
    route.fulfill({ json: ["BIAT"] }),
  );
  await page.route("**/market-api/stock/*/chart?*", (route) =>
    route.fulfill({ json: figure }),
  );
  await page.route("**/market-api/stock/*/news", (route) =>
    route.fulfill({ json: { articles: [], unavailable: false } }),
  );
  await page.route(/\/market-api\/stock\/BIAT$/, (route) =>
    route.fulfill({
      json: {
        columns: ["date", "cloture", "volume"],
        data: rows,
        forecast: {
          implied_price: 103.2,
          predicted_log_return: 0.014,
          as_of: "2026-10-12",
          forecast_horizon_observations: 1,
          horizon_basis: "observed_bars_unverified_sessions",
          price_basis: "unverified",
          model_version: "test-lstm-h1",
        },
      },
    }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toBeVisible();
  available = true;
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.locator(".js-plotly-plot")).toBeVisible();
  expect(
    await page
      .locator(".brand img")
      .evaluate(
        (node: HTMLImageElement) => node.complete && node.naturalWidth > 0,
      ),
  ).toBe(true);
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/overview-dark.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/overview-light.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Analyze BIAT" }).click();
  const forecast = page.locator(".forecast-panel");
  await expect(forecast.getByText("103.2", { exact: true })).toBeVisible();
  await expect(forecast.locator(".js-plotly-plot")).toBeVisible();
  const traces = await forecast
    .locator(".js-plotly-plot")
    .evaluate((node) =>
      (
        node as HTMLElement & { data: { name: string; y: number[] }[] }
      ).data.map((t) => ({ name: t.name, count: t.y.length })),
    );
  expect(traces).toEqual([
    { name: "Observed close", count: 12 },
    { name: "LSTM estimate", count: 1 },
  ]);
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/forecast-success.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 820, height: 1180 });
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/forecast-tablet.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/forecast-mobile.png",
    fullPage: true,
  });
});

test("market keeps history when forecasts fail and changes chart grouping", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/market-api/market", (route) =>
    route.fulfill({
      json: [
        {
          symbol: "BIAT",
          close: 102,
          change_pct: 2,
          volume: 30,
          as_of: "2026-10-02",
        },
        {
          symbol: "AL",
          close: 95,
          change_pct: -1,
          volume: 20,
          as_of: "2026-10-02",
        },
      ],
    }),
  );
  await page.route("**/market-api/stock/*/news", (route) =>
    route.fulfill({
      json: {
        articles: [
          {
            title: "Company earnings update",
            link: "https://example.com/news",
            publisher: "example.com",
            date: "2026-10-01",
          },
        ],
        unavailable: false,
        checked_at: "2026-10-02",
      },
    }),
  );
  await page.route("**/market-api/companies", (route) =>
    route.fulfill({ json: ["BIAT", "AL"] }),
  );
  await page.route(/\/market-api\/stock\/[^/?]+$/, (route) =>
    route.fulfill({
      json: {
        columns: ["date", "cloture", "volume"],
        data: [
          { date: "2026-10-01", cloture: 100, volume: 20 },
          { date: "2026-10-02", cloture: 102, volume: 30 },
        ],
        forecast_error: "Unavailable",
      },
    }),
  );
  await page.route("**/market-api/stock/*/chart?*", (route) =>
    route.fulfill({
      json: {
        data: [
          {
            type: "scatter",
            x: ["2026-10-01", "2026-10-02"],
            y: [100, 102],
            name: "Close",
          },
        ],
        layout: {
          paper_bgcolor: "#101722",
          plot_bgcolor: "#101722",
          font: { color: "#b7c4d6" },
        },
      },
    }),
  );
  await page.goto("/#market");
  await expect(page.getByRole("link", { name: "View BIAT" })).toBeVisible();
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/market-list.png",
    fullPage: true,
    animations: "disabled",
  });
  await page.getByLabel("Search stocks").fill("BIAT");
  for (const width of [820, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    await page.locator(".table-scroll").evaluate((node) => {
      node.scrollLeft = node.scrollWidth;
    });
    await expect(
      page.getByRole("link", { name: "View BIAT", exact: true }),
    ).toBeInViewport();
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(page.getByRole("link", { name: "View AL" })).toHaveCount(0);
  await page.getByRole("link", { name: "View BIAT" }).click();
  await expect(page).toHaveURL(/#market\/BIAT$/);
  await expect(page.getByText("BIAT / Last close")).toBeVisible();
  await expect(
    page.getByText(
      "The deployed forecast is temporarily unavailable. History is still available.",
    ),
  ).toBeVisible();
  await expect(page.locator(".js-plotly-plot")).toBeVisible();
  const monthly = page.waitForRequest(
    (r) => r.url().includes("/chart?") && r.url().includes("interval=Month"),
  );
  await page.getByLabel("Bar interval").selectOption("Month");
  await monthly;
  await page.getByLabel("Stock", { exact: true }).selectOption("AL");
  await expect(page.getByText("AL / Last close")).toBeVisible();
  await page.getByText("Historical data / export").click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download CSV" }).click();
  expect((await download).suggestedFilename()).toBe("AL-history.csv");
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator(".js-plotly-plot .main-svg").first()).toHaveCSS(
    "background-color",
    "rgb(255, 255, 255)",
  );
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/market-desktop.png",
    animations: "disabled",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page
        .locator(".js-plotly-plot .main-svg")
        .first()
        .evaluate((node) => node.getBoundingClientRect().width),
    )
    .toBeLessThanOrEqual(390);
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  expect(errors).toEqual([]);
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(
    page.getByRole("link", { name: "Company earnings update" }),
  ).toBeVisible();
  let turns = 0;
  await page.route("**/assistant-api/chat/stream", (route) => {
    turns++;
    expect(route.request().postDataJSON().question).toContain("AL");
    return route.fulfill({
      contentType: "application/x-ndjson",
      body:
        JSON.stringify({
          type: "result",
          text: "Here is the AL news context.",
        }) + "\n",
    });
  });
  await page
    .getByRole("button", {
      name: "What recent news could affect AL? Include dated sources.",
    })
    .click();
  await expect(page).toHaveURL(/#chat$/);
  await expect(page.getByText("Here is the AL news context.")).toBeVisible();
  expect(turns).toBe(1);
});

test("chat renders activity, sources and charts and preserves conversations while navigating", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/assistant-api/chat/stream", async (route) => {
    expect(route.request().postDataJSON().question).toContain("BIAT");
    await route.fulfill({
      contentType: "application/x-ndjson",
      body:
        [
          {
            type: "progress",
            id: "tool1",
            label: "Comparing stocks",
            status: "running",
          },
          {
            type: "progress",
            id: "tool1",
            label: "Comparing stocks",
            status: "completed",
          },
          {
            type: "result",
            text: "BIAT and AL are shown below.",
            sources: [
              { title: "Market source", link: "https://example.com/news" },
            ],
            charts: [
              {
                title: "Stock comparison",
                figure: {
                  data: [
                    { type: "scatter", x: [1, 2], y: [100, 103], name: "BIAT" },
                    { type: "scatter", x: [1, 2], y: [100, 101], name: "AL" },
                  ],
                  layout: {},
                },
              },
            ],
          },
        ]
          .map((e) => JSON.stringify(e))
          .join("\n") + "\n",
    });
  });
  await page.goto("/#chat");
  await page.screenshot({
    path: "test-results/chat-welcome.png",
    fullPage: true,
  });
  await expect(
    page.getByRole("button", { name: "Market workspace", exact: true }),
  ).toHaveCount(0);
  await page
    .getByLabel("Message TuniScope", { exact: true })
    .fill("Compare BIAT and AL");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByText("BIAT and AL are shown below.")).toBeVisible();
  await expect(page.getByText(/Response time:/)).toBeVisible();
  await page.getByText("Research activity", { exact: true }).click();
  await expect(page.getByText("Comparing stocks · completed")).toBeVisible();
  await page.getByText("Sources (1)").click();
  await expect(
    page.getByRole("link", { name: "Market source" }),
  ).toHaveAttribute("href", "https://example.com/news");
  await expect(page.locator(".js-plotly-plot .legendtext")).toHaveCount(2);
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/chat-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 820, height: 1180 });
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/chat-tablet.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(page.getByText("What can I help you research?")).toBeVisible();
  await page
    .getByRole("button", { name: "Compare BIAT and AL", exact: true })
    .click();
  await expect(page.getByText("BIAT and AL are shown below.")).toBeVisible();
  await page
    .getByRole("button", { name: "StockWise home", exact: true })
    .click();
  await page.getByRole("button", { name: "TuniScope", exact: true }).click();
  await expect(page.getByText("BIAT and AL are shown below.")).toBeVisible();
  await page
    .getByRole("button", { name: "Close sidebar", exact: true })
    .click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page
        .locator(".js-plotly-plot .main-svg")
        .first()
        .evaluate((node) => node.getBoundingClientRect().width),
    )
    .toBeLessThanOrEqual(390);
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/chat-mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator(".js-plotly-plot .main-svg").first()).toHaveCSS(
    "background-color",
    "rgb(255, 255, 255)",
  );
  await settleCharts(page);
  await page.screenshot({
    path: "test-results/chat-mobile-light.png",
    animations: "disabled",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Open sidebar", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Close sidebar", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(page.locator(".chat-sidebar")).toContainText(
    "Compare BIAT and AL",
  );
  expect(
    await page
      .locator(".chat-sidebar")
      .evaluate((node) => node.contains(document.activeElement)),
  ).toBe(true);
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "Open sidebar", exact: true }),
  ).toBeFocused();
  expect(errors).toEqual([]);
});
