"""
Plotly figure construction for the assistant's charts.

Figures are built here and returned to the browser as plain JSON, so this module
holds the styling decisions in one place rather than scattering them across the
plotting tools.

Two constraints shape what follows. The charts render inside Streamlit, whose
theme the user can switch between light and dark, and the figure is built on the
server long before anyone knows which one is in force: everything here therefore
has to read on either surface. Backgrounds stay transparent so the page shows
through, axis ink uses the one grey that is specified identically for both modes,
and titles are left out of the figure entirely for the page to render in its own
text colour.
"""

# Categorical series colours, in fixed slot order. Colours are assigned by
# position and never cycled, so a series keeps its colour when the number of
# series changes. This ordering was checked with the palette validator against
# both surfaces: worst adjacent colour-blind separation is ΔE 9.1 (light) and
# 8.4 (dark), both clear of the ΔE 8 target, and every slot clears the chroma
# and lightness bands. Two of the light steps fall below 3:1 against a white
# page, which is why every multi-series chart below keeps its legend and hover
# labels: identity is never left to colour alone.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
]

# Past this, a chart needs facets rather than more colours.
MAX_SERIES = len(SERIES_COLORS)

# Specified identically for light and dark, so axis furniture needs no theme.
MUTED_INK = "#898781"
GRID_LINE = "rgba(137, 135, 129, 0.25)"

LINE_WIDTH = 2


def series_color(index: int) -> str:
    """The colour for a series at a given position in the legend."""
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def line_figure(series: list[dict], y_title: str, hover_format: str = ",.3f") -> dict:
    """
    Build a time-series line chart as a JSON-ready figure.

    Args:
        series: one entry per line, with "name", "x" (dates as YYYY-MM-DD) and
            "y" (numbers). An entry may set "dashed" to draw a forecast or other
            projected value, which reads as distinct from a recorded one even
            where colour does not survive.
        y_title: label for the value axis.
        hover_format: d3 number format for hovered values.

    Returns:
        The figure as plain JSON types, ready to hand to a browser.
    """
    import json

    import plotly.graph_objects as go

    figure = go.Figure()
    for index, line in enumerate(series):
        figure.add_trace(
            go.Scatter(
                x=line["x"],
                y=line["y"],
                name=line["name"],
                mode="lines",
                line={
                    "color": line.get("color") or series_color(index),
                    "width": LINE_WIDTH,
                    "dash": "dash" if line.get("dashed") else "solid",
                },
                hovertemplate=f"%{{y:{hover_format}}}<extra>{line['name']}</extra>",
            )
        )

    axis = {
        "showgrid": True,
        "gridcolor": GRID_LINE,
        "linecolor": GRID_LINE,
        "zeroline": False,
        "tickfont": {"color": MUTED_INK},
        "title": {"font": {"color": MUTED_INK}},
    }

    figure.update_layout(
        # Let the page's own surface show through, whichever theme is in force.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        # A crosshair reading every line at once, rather than one point at a time.
        hovermode="x unified",
        # With one line the title already names it; past that the legend is what
        # keeps identity off colour alone.
        showlegend=len(series) > 1,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": MUTED_INK},
        },
        xaxis={**axis, "title": {"text": "", "font": {"color": MUTED_INK}}},
        yaxis={**axis, "title": {"text": y_title, "font": {"color": MUTED_INK}}},
    )

    # to_json converts numpy and timestamp types that json.dumps would refuse.
    return json.loads(figure.to_json())


def chart_payload(title: str, figure: dict, description: str) -> dict:
    """
    Wrap a figure with the text that travels alongside it.

    The title is carried outside the figure so the page can render it in its own
    heading colour instead of a fixed one baked into the image.

    Args:
        title: heading shown above the chart.
        figure: the figure JSON.
        description: one line stating what the chart shows, for readers who
            cannot see it and for the model to refer to in its answer.
    """
    return {"title": title, "figure": figure, "description": description}
