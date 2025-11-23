#!/usr/bin/env python3
"""
Global Terrorism interactive dashboard powered by Bokeh.

Run with:
    bokeh serve --show terrorism_dashboard.py
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.events import DocumentReady
from bokeh.models import (
    BasicTicker,
    Button,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    FactorRange,
    HoverTool,
    LabelSet,
    LinearAxis,
    LinearColorMapper,
    MultiSelect,
    NumberFormatter,
    Range1d,
    RangeSlider,
    Select,
    TableColumn,
    WMTSTileSource,
)
from bokeh.palettes import Turbo256, Category10, Category20
from bokeh.plotting import figure
from bokeh.transform import cumsum

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "globalterrorismdb_0718dist.zip"
TABLEAU_IFRAME_URL = (
    "https://public.tableau.com/views/same_17638730129700/1"
    "?:embed=yes&:showVizHome=no&:tabs=no&:toolbar=yes&:animate_transition=yes"
    "&:display_static_image=yes&:display_spinner=yes&:display_overlay=yes&:display_count=yes&:language=en-US"
    "&:device=desktop&:size=1200,850"
)
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CARD_STYLE = {
    "background-color": "#ffffff",
    "padding": "12px 14px",
    "border": "1px solid #e0e7f1",
    "border-radius": "10px",
    "box-shadow": "0 2px 6px rgba(15, 23, 42, 0.08)",
}
PERIOD_BINS = [1969, 1991, 2001, 2010, 2017]
PERIOD_LABELS = [
    "Late Cold War (1970-1991)",
    "Post-Cold War & 9/11 (1992-2001)",
    "Early War on Terror (2002-2010)",
    "Arab Spring / ISIS (2011-2017)",
]
FOCUS_REGIONS = ["Middle East & North Africa", "South Asia", "Sub-Saharan Africa", "South America"]
RECENT_YEAR_THRESHOLD = 2008


def to_mercator(lat: pd.Series, lon: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Convert latitude/longitude to Web Mercator."""
    k = 6378137
    x = lon * (math.pi / 180) * k
    y = np.log(np.tan((90 + lat) * math.pi / 360)) * k
    return x, y


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load and enrich the dataset."""
    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
    df["imonth"] = df["imonth"].replace(0, 1)
    df["iday"] = df["iday"].replace(0, 1)
    df["event_date"] = pd.to_datetime(
        dict(year=df["iyear"], month=df["imonth"], day=df["iday"]), errors="coerce"
    )
    df["nkill"] = df["nkill"].fillna(0)
    df["nwound"] = df["nwound"].fillna(0)
    df["casualties"] = df["nkill"] + df["nwound"]
    df["success_flag"] = df["success"].map({1: "Successful", 0: "Failed"})
    df["suicide_flag"] = df["suicide"].map({1: "Suicide", 0: "Non-suicide"})
    df["region_txt"] = df["region_txt"].fillna("Unknown region")
    df["attacktype1_txt"] = df["attacktype1_txt"].fillna("Unknown attack")
    df["targtype1_txt"] = df["targtype1_txt"].fillna("Unknown target")
    df["country_txt"] = df["country_txt"].fillna("Unknown country")
    df["city"] = df["city"].fillna("Unknown city")
    df["weaptype1_txt"] = (
        df["weaptype1_txt"]
        .fillna("Unknown weapon")
        .replace(
            "Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)",
            "Vehicle",
        )
    )
    df["gname"] = df["gname"].fillna("Unknown group")
    df["month_num"] = df["imonth"].clip(1, 12).astype(int)
    df["month_name"] = pd.Categorical(
        [MONTH_ORDER[m - 1] for m in df["month_num"]],
        categories=MONTH_ORDER,
        ordered=True,
    )
    df["decade"] = ((df["iyear"] // 10) * 10).astype(int)
    df["period"] = pd.cut(df["iyear"], bins=PERIOD_BINS, labels=PERIOD_LABELS)
    df["recent_flag"] = df["iyear"] >= RECENT_YEAR_THRESHOLD
    return df


data = load_data()

# Widget setup
year_min, year_max = int(data["iyear"].min()), int(data["iyear"].max())
regions = sorted(data["region_txt"].dropna().unique().tolist())
attack_types = sorted(data["attacktype1_txt"].dropna().unique().tolist())
target_types = sorted(data["targtype1_txt"].dropna().unique().tolist())
hotspot_regions = [region for region in FOCUS_REGIONS if region in regions] or regions[:5]
top_weapon_types = data["weaptype1_txt"].value_counts().head(5).index.tolist() or ["Unknown weapon"]
top_orgs = (
    data.loc[data["gname"].ne("Unknown group"), "gname"].value_counts().head(5).index.tolist()
    or ["Unknown group"]
)
top_target_types = data["targtype1_txt"].value_counts().head(8).index.tolist() or ["Unknown target"]
region_palette = Category20[20]
REGION_COLOR_MAP = {region: region_palette[i % len(region_palette)] for i, region in enumerate(regions)}

year_slider = RangeSlider(
    title="Year range",
    start=year_min,
    end=year_max,
    value=(max(year_min, 1990), year_max),
    step=1,
)

region_select = MultiSelect(
    title="Regions",
    options=regions,
    value=regions,
    size=min(len(regions), 12),
)

attack_select = MultiSelect(
    title="Attack types",
    options=attack_types,
    value=attack_types,
    size=min(len(attack_types), 12),
)

target_select = MultiSelect(
    title="Target categories",
    options=target_types,
    value=target_types,
    size=min(len(target_types), 12),
)

fatality_slider = RangeSlider(
    title="Fatalities per event",
    start=0,
    end=int(data["nkill"].max()),
    value=(0, int(np.quantile(data["nkill"], 0.95))),
    step=1,
)

casualty_slider = RangeSlider(
    title="Total casualties (killed + wounded)",
    start=0,
    end=int(data["casualties"].max()),
    value=(0, int(np.quantile(data["casualties"], 0.95))),
    step=1,
)

success_select = Select(
    title="Operation outcome",
    value="All",
    options=["All", "Successful", "Failed"],
)

suicide_select = Select(
    title="Suicide involvement",
    value="All",
    options=["All", "Suicide", "Non-suicide"],
)

timeline_metric = Select(
    title="Timeline metric",
    value="Incidents",
    options=["Incidents", "Fatalities", "Wounded", "Casualties"],
)

highlight_region_select = Select(
    title="Highlight region",
    value="None",
    options=["None"] + regions,
)

hotspot_region_select = Select(
    title="City hotspot focus",
    value=hotspot_regions[0] if hotspot_regions else "All",
    options=hotspot_regions or ["All"],
)

reset_button = Button(label="Reset filters", button_type="primary")


# Data sources
map_source = ColumnDataSource(
    data=dict(
        mercator_x=[],
        mercator_y=[],
        country_txt=[],
        region_txt=[],
        city=[],
        incidents=[],
        fatalities=[],
        wounded=[],
        casualties=[],
        years=[],
        size=[],
        color=[],
        alpha=[],
    )
)

timeline_source = ColumnDataSource(data=dict(year=[], incidents=[], fatalities=[], wounded=[], casualties=[]))

timeline_events_source = ColumnDataSource(data=dict(year=[], metric=[], text=[]))

country_source = ColumnDataSource(data=dict(country_txt=[], incidents=[], casualties=[]))

attack_source = ColumnDataSource(data=dict(attacktype1_txt=[], incidents=[], casualties=[]))

target_source = ColumnDataSource(data=dict(targtype1_txt=[], incidents=[], casualties=[]))

season_source = ColumnDataSource(data=dict(month=[], incidents=[]))

heatmap_source = ColumnDataSource(data=dict(decade=[], region=[], casualties=[], color=[]))

region_trend_source = ColumnDataSource(data=dict(xs=[], ys=[], legend=[], color=[]))

weapon_source = ColumnDataSource(data=dict(weaptype1_txt=[], incidents=[], casualties=[]))

success_source = ColumnDataSource(
    data=dict(
        category=[],
        angle=[],
        color=[],
        label=[],
        incidents=[],
        percent=[],
        label_x=[],
        label_y=[],
        label_short=[],
    )
)

region_stack_source = ColumnDataSource(data=dict(year=[], **{region: [] for region in regions}))
region_highlight_source = ColumnDataSource(data=dict(year=[], value=[]))

period_source = ColumnDataSource(data=dict(period=[], events=[], casualties=[]))

hotspot_source = ColumnDataSource(
    data=dict(city=[], events=[], avg_casualties=[], label_x=[], label_text=[])
)

attack_target_source = ColumnDataSource(data=dict(attack=[], target=[], incidents=[]))

weapon_share_source = ColumnDataSource(data=dict(year=[], **{weapon: [] for weapon in top_weapon_types}))

org_trend_source = ColumnDataSource(data=dict(year=[], **{org: [] for org in top_orgs}))

severity_source = ColumnDataSource(data=dict(target=[], killed=[], wounded=[]))

table_source = ColumnDataSource(
    data=dict(
        event_date=[],
        country_txt=[],
        city=[],
        attacktype1_txt=[],
        targtype1_txt=[],
        nkill=[],
        nwound=[],
        casualties=[],
    )
)


# Figures
tile_provider = WMTSTileSource(
    url="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{Z}/{X}/{Y}.png"
)

map_fig = figure(
    title="Geospatial distribution of incidents",
    height=460,
    width=780,
    sizing_mode="stretch_width",
    x_axis_type="mercator",
    y_axis_type="mercator",
    tools="pan,wheel_zoom,reset",
    active_scroll="wheel_zoom",
)
map_fig.add_tile(tile_provider)
map_renderer = map_fig.scatter(
    x="mercator_x",
    y="mercator_y",
    size="size",
    source=map_source,
    fill_color="color",
    fill_alpha="alpha",
    line_color="#1f2d44",
    line_alpha=0.6,
)
map_hover = HoverTool(
    renderers=[map_renderer],
    tooltips=[
        ("Location", "@city, @country_txt"),
        ("Region", "@region_txt"),
        ("Incidents", "@incidents"),
        ("Casualties", "@casualties"),
        ("Killed", "@fatalities"),
        ("Wounded", "@wounded"),
        ("Years covered", "@years"),
    ],
)
map_fig.add_tools(map_hover)

timeline_fig = figure(
    title="Timeline of selected metric",
    height=450,
    width=780,
    sizing_mode="stretch_width",
    x_axis_label="Year",
    y_axis_label="Incidents",
    tools="pan,xwheel_zoom,reset,save",
)
timeline_renderer = timeline_fig.line(
    x="year",
    y="metric",
    source=ColumnDataSource(data=dict(year=[], metric=[])),
    line_width=3,
    color="#c94d6d",
)
timeline_hover = HoverTool(
    renderers=[timeline_renderer],
    tooltips=[("Year", "@year"), ("Value", "@metric{0,0}")],
    mode="vline",
)
timeline_fig.add_tools(timeline_hover)
timeline_events_renderer = timeline_fig.scatter(
    x="year",
    y="metric",
    size=10,
    color="#1d3557",
    alpha=0.85,
    source=timeline_events_source,
)
timeline_labels = LabelSet(
    x="year",
    y="metric",
    text="text",
    level="overlay",
    source=timeline_events_source,
    text_font_size="9pt",
    text_color="#1d3557",
    y_offset=6,
    text_align="center",
)
timeline_fig.add_layout(timeline_labels)

country_fig = figure(
    title="Top countries by casualties",
    height=340,
    width=520,
    sizing_mode="stretch_width",
    y_range=FactorRange(),
    x_axis_label="Casualties",
    toolbar_location=None,
)
country_renderer = country_fig.hbar(
    y="country_txt",
    right="casualties",
    height=0.55,
    source=country_source,
    color="#0d77b6",
)
country_labels = LabelSet(
    x="casualties",
    y="country_txt",
    text="casualties",
    source=country_source,
    text_font_size="9pt",
    x_offset=5,
    text_baseline="middle",
)
country_fig.add_layout(country_labels)

attack_fig = figure(
    title="Attack-type mix",
    height=340,
    width=520,
    sizing_mode="stretch_width",
    x_range=FactorRange(),
    x_axis_label="Attack type",
    y_axis_label="Incidents",
    toolbar_location=None,
)
attack_renderer = attack_fig.vbar(
    x="attacktype1_txt",
    top="incidents",
    width=0.8,
    source=attack_source,
    color="#f18f01",
)
attack_labels = LabelSet(
    x="attacktype1_txt",
    y="incidents",
    text="incidents",
    source=attack_source,
    text_font_size="9pt",
    y_offset=5,
    text_align="center",
)
attack_fig.add_layout(attack_labels)

target_fig = figure(
    title="Top target categories",
    height=340,
    width=520,
    sizing_mode="stretch_width",
    y_range=FactorRange(),
    x_axis_label="Casualties",
    toolbar_location=None,
)
target_fig.hbar(
    y="targtype1_txt",
    right="casualties",
    height=0.55,
    source=target_source,
    color="#7e57c2",
)
target_labels = LabelSet(
    x="casualties",
    y="targtype1_txt",
    text="casualties",
    source=target_source,
    text_font_size="9pt",
    x_offset=5,
    text_baseline="middle",
)
target_fig.add_layout(target_labels)

season_fig = figure(
    title="Monthly seasonality of incidents",
    height=280,
    width=620,
    sizing_mode="stretch_width",
    x_range=MONTH_ORDER,
    y_axis_label="Incidents",
    toolbar_location=None,
)
season_fig.vbar(x="month", top="incidents", width=0.8, source=season_source, color="#2a9d8f")
season_labels = LabelSet(
    x="month",
    y="incidents",
    text="incidents",
    source=season_source,
    text_font_size="9pt",
    y_offset=5,
    text_align="center",
)
season_fig.add_layout(season_labels)

heatmap_fig = figure(
    title="Regional casualty intensity by decade",
    height=360,
    width=780,
    sizing_mode="stretch_width",
    x_range=FactorRange(),
    y_range=FactorRange(),
    toolbar_location=None,
    tooltips=[("Region", "@region"), ("Decade", "@decade"), ("Casualties", "@casualties{0,0}")],
)
heatmap_fig.rect(
    x="decade",
    y="region",
    width=0.9,
    height=0.9,
    fill_color="color",
    line_color=None,
    fill_alpha=0.85,
    source=heatmap_source,
)

region_trend_fig = figure(
    title="Regional casualty trends",
    height=320,
    width=620,
    sizing_mode="stretch_width",
    x_axis_label="Year",
    y_axis_label="Casualties",
    tools="pan,xwheel_zoom,reset,save",
)
region_trend_renderer = region_trend_fig.multi_line(
    xs="xs",
    ys="ys",
    color="color",
    legend_field="legend",
    line_width=3,
    source=region_trend_source,
)
region_trend_hover = HoverTool(
    renderers=[region_trend_renderer],
    tooltips=[("Region", "@legend"), ("Year", "$x{0}"), ("Casualties", "$y{0,0}")],
    line_policy="nearest",
)
region_trend_fig.add_tools(region_trend_hover)
region_trend_fig.legend.location = "top_left"
region_trend_fig.legend.click_policy = "hide"

weapon_fig = figure(
    title="Weapon types by incidents",
    height=320,
    width=520,
    sizing_mode="stretch_width",
    x_axis_label="Weapon type",
    y_axis_label="Incidents",
    x_range=FactorRange(),
    toolbar_location=None,
    min_border_left=50,
)
weapon_fig.vbar(
    x="weaptype1_txt",
    top="incidents",
    width=0.8,
    source=weapon_source,
    color="#ef476f",
)
weapon_labels = LabelSet(
    x="weaptype1_txt",
    y="incidents",
    text="incidents",
    source=weapon_source,
    text_font_size="9pt",
    y_offset=5,
    text_align="center",
)
weapon_fig.add_layout(weapon_labels)

success_pie_fig = figure(
    title="Mission outcome split",
    height=350,
    width=350,
    match_aspect=True,
    toolbar_location=None,
    tools="hover",
    tooltips="@label",
    sizing_mode="fixed",
)
success_pie_fig.annular_wedge(
    x=0,
    y=0,
    inner_radius=0,
    outer_radius=0.9,
    start_angle=cumsum("angle", include_zero=True),
    end_angle=cumsum("angle"),
    line_color="white",
    fill_color="color",
    legend_field="category",
    source=success_source,
)
success_labels = LabelSet(
    x="label_x",
    y="label_y",
    text="label_short",
    source=success_source,
    text_color="white",
    text_font_size="10pt",
    text_align="center",
    text_baseline="middle",
)
success_pie_fig.add_layout(success_labels)
success_pie_fig.axis.visible = False
success_pie_fig.grid.grid_line_color = None
success_pie_fig.legend.orientation = "horizontal"
success_pie_fig.legend.location = "bottom_center"
success_pie_fig.min_border = 10

region_stack_fig = figure(
    title="Regional share of incidents over time",
    height=360,
    width=780,
    sizing_mode="stretch_width",
    x_axis_label="Year",
    y_axis_label="Incidents (stacked)",
    tools="pan,xwheel_zoom,reset,save",
)
stack_colors = [REGION_COLOR_MAP[r] for r in regions]
region_stack_fig.varea_stack(
    stackers=regions,
    x="year",
    color=stack_colors,
    legend_label=regions,
    source=region_stack_source,
)
region_highlight_renderer = region_stack_fig.line(
    x="year",
    y="value",
    color="#000000",
    line_width=4,
    alpha=0.9,
    source=region_highlight_source,
)
region_stack_fig.legend.orientation = "horizontal"
region_stack_fig.legend.location = "top_center"
region_stack_fig.legend.click_policy = "mute"
region_stack_fig.yaxis[0].formatter.use_scientific = False

period_fig = figure(
    title="Historical period comparison",
    height=320,
    width=520,
    x_range=PERIOD_LABELS,
    toolbar_location=None,
)
period_fig.yaxis.axis_label = "Incidents"
period_fig.extra_y_ranges = {"casualties": Range1d(start=0, end=1)}
period_fig.add_layout(LinearAxis(y_range_name="casualties", axis_label="Casualties"), "right")
period_fig.vbar(x="period", top="events", width=0.4, source=period_source, color="#2ca02c")
period_fig.line(
    x="period",
    y="casualties",
    y_range_name="casualties",
    line_width=3,
    color="#ff7f0e",
    source=period_source,
)
period_line_renderer = period_fig.renderers[-1]
period_hover = HoverTool(
    renderers=[period_line_renderer],
    tooltips=[("Period", "@period"), ("Casualties", "@casualties{0,0}"), ("Events", "@events{0,0}")],
    mode="vline",
)
period_fig.add_tools(period_hover)
period_labels = LabelSet(
    x="period",
    y="events",
    text="events",
    source=period_source,
    text_font_size="9pt",
    y_offset=5,
    text_align="center",
)
period_fig.add_layout(period_labels)
period_fig.xaxis.major_label_orientation = math.radians(10)

hotspot_fig = figure(
    title="Regional city hotspots",
    height=320,
    width=520,
    x_axis_label="Incidents",
    y_range=FactorRange(),
    toolbar_location=None,
)
hotspot_fig.hbar(
    y="city",
    right="events",
    height=0.55,
    source=hotspot_source,
    color="#d45087",
)
hotspot_labels = LabelSet(
    x="label_x",
    y="city",
    text="label_text",
    source=hotspot_source,
    text_font_size="9pt",
    text_color="#4b5563",
    text_baseline="middle",
)
hotspot_fig.add_layout(hotspot_labels)

attack_target_color_mapper = LinearColorMapper(palette=Turbo256, low=0, high=1)
attack_target_fig = figure(
    title="Attack vs target matrix",
    height=360,
    width=780,
    x_range=FactorRange(),
    y_range=FactorRange(),
    toolbar_location=None,
    tooltips=[("Attack", "@attack"), ("Target", "@target"), ("Incidents", "@incidents{0,0}")],
)
attack_target_fig.rect(
    x="target",
    y="attack",
    width=0.9,
    height=0.9,
    source=attack_target_source,
    fill_color={"field": "incidents", "transform": attack_target_color_mapper},
    line_color=None,
)
attack_target_color_bar = ColorBar(
    color_mapper=attack_target_color_mapper,
    ticker=BasicTicker(desired_num_ticks=8),
    label_standoff=8,
    location=(0, 0),
)
attack_target_fig.add_layout(attack_target_color_bar, "right")
attack_target_fig.xaxis.major_label_orientation = math.radians(35)

weapon_share_colors = Category20[20]
weapon_share_fig = figure(
    title="Weapon share over time",
    height=320,
    width=780,
    x_axis_label="Year",
    y_axis_label="Share of incidents",
    y_range=(0, 1),
    tools="pan,xwheel_zoom,reset,save",
)
weapon_share_fig.varea_stack(
    stackers=top_weapon_types,
    x="year",
    color=weapon_share_colors[: len(top_weapon_types)],
    legend_label=top_weapon_types,
    source=weapon_share_source,
)
weapon_share_fig.legend.location = "top_left"
weapon_share_fig.legend.click_policy = "hide"

org_colors = Category20[20]
org_trend_fig = figure(
    title="Top organizations incident volume",
    height=320,
    width=780,
    x_axis_label="Year",
    y_axis_label="Incidents",
    tools="pan,xwheel_zoom,reset,save",
)
org_trend_fig.varea_stack(
    stackers=top_orgs,
    x="year",
    color=org_colors[: len(top_orgs)],
    legend_label=top_orgs,
    source=org_trend_source,
)
org_trend_fig.legend.location = "top_left"
org_trend_fig.legend.click_policy = "hide"

target_severity_fig = figure(
    title="Target severity (killed vs wounded)",
    height=320,
    width=780,
    x_range=FactorRange(),
    y_axis_label="People",
    toolbar_location=None,
)
target_severity_fig.vbar_stack(
    stackers=["killed", "wounded"],
    x="target",
    color=["#d73027", "#fc8d59"],
    source=severity_source,
    legend_label=["Killed", "Wounded"],
)
target_severity_fig.xaxis.major_label_orientation = math.radians(25)
target_severity_fig.legend.location = "top_right"

attack_fig.xaxis.major_label_orientation = math.radians(35)
weapon_fig.xaxis.major_label_orientation = math.radians(35)
season_fig.xaxis.major_label_orientation = math.radians(20)
timeline_fig.min_border_left = 55
region_trend_fig.min_border_left = 60
region_stack_fig.min_border_left = 60
for fig in [country_fig, attack_fig, target_fig, weapon_fig, region_trend_fig, heatmap_fig, region_stack_fig]:
    fig.title.text_font_size = "13pt"
    fig.xaxis.major_label_text_font_size = "10pt"
    fig.yaxis.major_label_text_font_size = "10pt"
    fig.xaxis.axis_label_text_font_size = "11pt"
    fig.yaxis.axis_label_text_font_size = "11pt"

summary_div = Div(text="", height=120, sizing_mode="stretch_width", styles={"font-size": "13px"})
tableau_embed_div = Div(
    text=f"""
    <div style="width:100%; text-align:center; margin:0 auto;">
        <iframe src="{TABLEAU_IFRAME_URL}"
                style="border:none; width:1200px; height:850px; display:inline-block;"
                loading="lazy" allowfullscreen>
        </iframe>
    </div>
    """,
    sizing_mode="stretch_width",
    styles={"width": "100%", "margin": "0 auto"},
)

columns = [
    TableColumn(field="event_date", title="Date"),
    TableColumn(field="country_txt", title="Country"),
    TableColumn(field="city", title="City"),
    TableColumn(field="attacktype1_txt", title="Attack type"),
    TableColumn(field="targtype1_txt", title="Target type"),
    TableColumn(field="nkill", title="Killed", formatter=NumberFormatter(format="0,0")),
    TableColumn(field="nwound", title="Wounded", formatter=NumberFormatter(format="0,0")),
    TableColumn(field="casualties", title="Casualties", formatter=NumberFormatter(format="0,0")),
]
table = DataTable(source=table_source, columns=columns, width=1180, height=280, index_position=None, selectable=True)


def _safe_quantile(series: pd.Series, q: float) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(np.quantile(clean, q))


def build_palette(values: pd.Series) -> List[str]:
    """Map numeric values onto Turbo256 palette."""
    if values.empty:
        return []
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return ["#ffcc00"] * len(values)
    scaled = ((values - vmin) / (vmax - vmin) * (len(Turbo256) - 1)).astype(int)
    return [Turbo256[i] for i in scaled]


def filter_data() -> pd.DataFrame:
    subset = data.copy()
    year_start, year_end = year_slider.value
    subset = subset[(subset["iyear"] >= year_start) & (subset["iyear"] <= year_end)]

    if region_select.value:
        subset = subset[subset["region_txt"].isin(region_select.value)]
    if attack_select.value:
        subset = subset[subset["attacktype1_txt"].isin(attack_select.value)]
    if target_select.value:
        subset = subset[subset["targtype1_txt"].isin(target_select.value)]

    fatality_min, fatality_max = fatality_slider.value
    subset = subset[(subset["nkill"] >= fatality_min) & (subset["nkill"] <= fatality_max)]

    casualty_min, casualty_max = casualty_slider.value
    subset = subset[(subset["casualties"] >= casualty_min) & (subset["casualties"] <= casualty_max)]

    if success_select.value != "All":
        subset = subset[subset["success_flag"] == success_select.value]

    if suicide_select.value != "All":
        subset = subset[subset["suicide_flag"] == suicide_select.value]

    return subset


def update_map(subset: pd.DataFrame) -> None:
    geo_df = subset.dropna(subset=["latitude", "longitude"])
    if geo_df.empty:
        map_source.data = {key: [] for key in map_source.data.keys()}
        map_source.data["size"] = []
        map_source.data["color"] = []
        return

    grouped = (
        geo_df.groupby(
            ["country_txt", "region_txt", "provstate", "city", "latitude", "longitude"], as_index=False
        )
        .agg(
            incidents=("eventid", "count"),
            fatalities=("nkill", "sum"),
            wounded=("nwound", "sum"),
            casualties=("casualties", "sum"),
            year_min=("iyear", "min"),
            year_max=("iyear", "max"),
        )
        .sort_values("casualties", ascending=False)
    )
    merc_x, merc_y = to_mercator(grouped["latitude"], grouped["longitude"])
    grouped["mercator_x"] = merc_x
    grouped["mercator_y"] = merc_y
    grouped["years"] = grouped.apply(
        lambda row: f"{int(row.year_min)}" + (f"–{int(row.year_max)}" if row.year_max != row.year_min else ""),
        axis=1,
    )

    max_casualty = max(grouped["casualties"].max(), 1)
    grouped["size"] = np.interp(grouped["casualties"], [0, max_casualty], [6, 36])
    grouped["color"] = build_palette(grouped["casualties"])
    selected_region = highlight_region_select.value
    if selected_region != "None":
        grouped["alpha"] = np.where(grouped["region_txt"] == selected_region, 0.85, 0.2)
    else:
        grouped["alpha"] = 0.55

    map_source.data = grouped.to_dict(orient="list")


def update_timeline(subset: pd.DataFrame) -> None:
    if subset.empty:
        timeline_source.data = {key: [] for key in timeline_source.data.keys()}
        timeline_renderer.data_source.data = dict(year=[], metric=[])
        timeline_events_source.data = dict(year=[], metric=[], text=[])
        return

    ts = (
        subset.groupby("iyear")
        .agg(
            incidents=("eventid", "count"),
            fatalities=("nkill", "sum"),
            wounded=("nwound", "sum"),
            casualties=("casualties", "sum"),
        )
        .reset_index()
        .sort_values("iyear")
    )
    timeline_source.data = ts.to_dict(orient="list")

    metric_map = {
        "Incidents": "incidents",
        "Fatalities": "fatalities",
        "Wounded": "wounded",
        "Casualties": "casualties",
    }
    metric_column = metric_map[timeline_metric.value]
    timeline_renderer.data_source.data = dict(year=ts["iyear"], metric=ts[metric_column])
    timeline_fig.yaxis.axis_label = timeline_metric.value
    update_timeline_events(subset, ts, metric_column)


def update_timeline_events(subset: pd.DataFrame, ts: pd.DataFrame, metric_column: str) -> None:
    if subset.empty or ts.empty:
        timeline_events_source.data = dict(year=[], metric=[], text=[])
        return

    year_metric_map = dict(zip(ts["iyear"], ts[metric_column]))
    top_events = (
        subset.sort_values("casualties", ascending=False)
        .head(8)
        .copy()
    )
    if top_events.empty:
        timeline_events_source.data = dict(year=[], metric=[], text=[])
        return

    max_metric = max(float(ts[metric_column].max()), 1.0)
    y_values = []
    labels = []
    for i, (_, row) in enumerate(top_events.iterrows()):
        base = year_metric_map.get(row["iyear"], 0.0)
        y_values.append(base)
        country = row.get("country_txt", "Unknown")
        city = row.get("city", "Unknown city")
        casualties = int(row.get("casualties", 0))
        labels.append(f"{country} · {city} ({casualties})")
    timeline_events_source.data = dict(
        year=top_events["iyear"].tolist(),
        metric=y_values,
        text=labels,
    )


def update_country_bar(subset: pd.DataFrame) -> None:
    if subset.empty:
        country_source.data = dict(country_txt=[], incidents=[], casualties=[])
        country_fig.y_range.factors = []
        return

    country_df = (
        subset.groupby("country_txt")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .reset_index()
        .sort_values("casualties", ascending=False)
        .head(12)
    )
    country_source.data = country_df.to_dict(orient="list")
    country_fig.y_range.factors = country_df["country_txt"].tolist()[::-1]


def update_attack_bar(subset: pd.DataFrame) -> None:
    if subset.empty:
        attack_source.data = dict(attacktype1_txt=[], incidents=[], casualties=[])
        attack_fig.x_range.factors = []
        return

    attack_df = (
        subset.groupby("attacktype1_txt")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .reset_index()
        .sort_values("incidents", ascending=False)
    )
    attack_source.data = attack_df.to_dict(orient="list")
    attack_fig.x_range.factors = attack_df["attacktype1_txt"].tolist()


def update_target_bar(subset: pd.DataFrame) -> None:
    if subset.empty:
        target_source.data = dict(targtype1_txt=[], incidents=[], casualties=[])
        target_fig.y_range.factors = []
        return

    target_df = (
        subset.groupby("targtype1_txt")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .reset_index()
        .sort_values("casualties", ascending=False)
        .head(12)
    )
    target_source.data = target_df.to_dict(orient="list")
    target_fig.y_range.factors = target_df["targtype1_txt"].tolist()[::-1]


def update_seasonality(subset: pd.DataFrame) -> None:
    if subset.empty:
        season_source.data = dict(month=[], incidents=[])
        return

    month_counts = (
        subset.groupby("month_name", observed=False)
        .agg(incidents=("eventid", "count"))
        .reindex(MONTH_ORDER, fill_value=0)
        .reset_index()
        .rename(columns={"month_name": "month"})
    )
    season_source.data = dict(month=month_counts["month"], incidents=month_counts["incidents"])


def update_heatmap(subset: pd.DataFrame) -> None:
    if subset.empty:
        heatmap_source.data = dict(decade=[], region=[], casualties=[], color=[])
        heatmap_fig.x_range.factors = []
        heatmap_fig.y_range.factors = []
        return

    heat_df = (
        subset.groupby(["region_txt", "decade"])
        .agg(casualties=("casualties", "sum"))
        .reset_index()
    )
    heat_df["decade_label"] = heat_df["decade"].astype(str)
    heat_df["color"] = build_palette(heat_df["casualties"])
    heatmap_source.data = dict(
        decade=heat_df["decade_label"],
        region=heat_df["region_txt"],
        casualties=heat_df["casualties"],
        color=heat_df["color"],
    )
    heatmap_fig.x_range.factors = sorted(heat_df["decade_label"].unique())
    heatmap_fig.y_range.factors = sorted(heat_df["region_txt"].unique())


def update_period_view(subset: pd.DataFrame) -> None:
    period_df = subset.dropna(subset=["period"])
    if period_df.empty:
        period_source.data = dict(period=[], events=[], casualties=[])
        period_fig.y_range.end = 1
        period_fig.extra_y_ranges["casualties"].end = 1
        return

    summary = (
        period_df.groupby("period", observed=False)
        .agg(events=("eventid", "count"), casualties=("casualties", "sum"))
        .reindex(PERIOD_LABELS)
        .fillna(0)
    )
    period_source.data = {
        "period": summary.index.tolist(),
        "events": summary["events"].tolist(),
        "casualties": summary["casualties"].tolist(),
    }
    period_fig.y_range.end = max(summary["events"].max() * 1.1, 1)
    period_fig.extra_y_ranges["casualties"].end = max(summary["casualties"].max() * 1.1, 1)


def update_region_stack(subset: pd.DataFrame) -> None:
    if subset.empty:
        region_stack_source.data = dict(year=[], **{region: [] for region in regions})
        region_highlight_source.data = dict(year=[], value=[])
        return

    stack_df = (
        subset.groupby(["iyear", "region_txt"])
        .agg(incidents=("eventid", "count"))
        .reset_index()
    )
    pivot = (
        stack_df.pivot(index="iyear", columns="region_txt", values="incidents")
        .reindex(sorted(stack_df["iyear"].unique()))
        .fillna(0)
    )
    data_dict = {"year": pivot.index.tolist()}
    for region in regions:
        data_dict[region] = pivot.get(region, pd.Series(0, index=pivot.index)).tolist()
    region_stack_source.data = data_dict

    selected_region = highlight_region_select.value
    if selected_region != "None" and selected_region in pivot.columns:
        region_highlight_source.data = dict(
            year=data_dict["year"],
            value=pivot[selected_region].tolist(),
        )
        region_highlight_renderer.glyph.line_color = REGION_COLOR_MAP.get(selected_region, "#000000")
    else:
        region_highlight_source.data = dict(year=[], value=[])


def update_region_trend(subset: pd.DataFrame) -> None:
    if subset.empty:
        region_trend_source.data = dict(xs=[], ys=[], legend=[], color=[])
        return

    region_totals = (
        subset.groupby("region_txt")
        .agg(casualties=("casualties", "sum"))
        .sort_values("casualties", ascending=False)
        .head(4)
    )
    selected_regions = region_totals.index.tolist()
    trend_df = (
        subset[subset["region_txt"].isin(selected_regions)]
        .groupby(["region_txt", "iyear"])
        .agg(casualties=("casualties", "sum"))
        .reset_index()
        .sort_values("iyear")
    )
    xs, ys, legends, colors = [], [], [], []
    palette = Category10[10]
    for idx, region in enumerate(selected_regions):
        region_data = trend_df[trend_df["region_txt"] == region]
        xs.append(region_data["iyear"].tolist())
        ys.append(region_data["casualties"].tolist())
        legends.append(region)
        colors.append(palette[idx % len(palette)])
    region_trend_source.data = dict(xs=xs, ys=ys, legend=legends, color=colors)


def update_hotspots(subset: pd.DataFrame) -> None:
    focus_region = hotspot_region_select.value
    region_df = subset[subset["region_txt"] == focus_region]
    if region_df.empty:
        hotspot_source.data = dict(city=[], events=[], avg_casualties=[], label_x=[], label_text=[])
        hotspot_fig.y_range.factors = []
        return

    top_cities = (
        region_df.groupby("city")
        .agg(events=("eventid", "count"), avg_casualties=("casualties", "mean"))
        .sort_values("events", ascending=False)
        .head(10)
    )
    max_events = max(top_cities["events"].max(), 1)
    hotspot_source.data = dict(
        city=top_cities.index.tolist(),
        events=top_cities["events"].tolist(),
        avg_casualties=top_cities["avg_casualties"].tolist(),
        label_x=(top_cities["events"] + max_events * 0.02).tolist(),
        label_text=[f"{avg:.1f} avg casualties" for avg in top_cities["avg_casualties"]],
    )
    hotspot_fig.y_range.factors = top_cities.index.tolist()[::-1]


def update_attack_target_matrix(subset: pd.DataFrame) -> None:
    if subset.empty:
        attack_target_source.data = dict(attack=[], target=[], incidents=[])
        attack_target_fig.x_range.factors = []
        attack_target_fig.y_range.factors = []
        attack_target_color_mapper.high = 1
        return

    matrix = (
        subset.groupby(["attacktype1_txt", "targtype1_txt"])
        .size()
        .reset_index(name="incidents")
    )
    attack_target_source.data = dict(
        attack=matrix["attacktype1_txt"].tolist(),
        target=matrix["targtype1_txt"].tolist(),
        incidents=matrix["incidents"].tolist(),
    )
    attack_target_fig.y_range.factors = sorted(matrix["attacktype1_txt"].unique().tolist())[::-1]
    attack_target_fig.x_range.factors = sorted(matrix["targtype1_txt"].unique().tolist())
    attack_target_color_mapper.high = max(matrix["incidents"].max(), 1)


def update_weapon_bar(subset: pd.DataFrame) -> None:
    if subset.empty:
        weapon_source.data = dict(weaptype1_txt=[], incidents=[], casualties=[])
        weapon_fig.x_range.factors = []
        return

    weapon_df = (
        subset.groupby("weaptype1_txt")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .reset_index()
        .sort_values("incidents", ascending=False)
        .head(10)
    )
    weapon_source.data = weapon_df.to_dict(orient="list")
    weapon_fig.x_range.factors = weapon_df["weaptype1_txt"].tolist()


def update_weapon_share(subset: pd.DataFrame) -> None:
    filtered = subset[subset["weaptype1_txt"].isin(top_weapon_types)]
    if filtered.empty:
        weapon_share_source.data = dict(year=[], **{weapon: [] for weapon in top_weapon_types})
        return

    weapon_counts = (
        filtered.groupby(["iyear", "weaptype1_txt"])
        .size()
        .reset_index(name="incidents")
    )
    pivot = (
        weapon_counts.pivot(index="iyear", columns="weaptype1_txt", values="incidents")
        .reindex(columns=top_weapon_types, fill_value=0)
        .sort_index()
    )
    totals = pivot.sum(axis=1).replace(0, 1)
    share = pivot.div(totals, axis=0)
    data_dict = {"year": share.index.tolist()}
    for weapon in top_weapon_types:
        data_dict[weapon] = share[weapon].tolist()
    weapon_share_source.data = data_dict


def update_org_trend(subset: pd.DataFrame) -> None:
    filtered = subset[subset["gname"].isin(top_orgs)]
    if filtered.empty:
        org_trend_source.data = dict(year=[], **{org: [] for org in top_orgs})
        return

    org_counts = (
        filtered.groupby(["iyear", "gname"])
        .size()
        .reset_index(name="incidents")
    )
    pivot = (
        org_counts.pivot(index="iyear", columns="gname", values="incidents")
        .reindex(columns=top_orgs, fill_value=0)
        .sort_index()
    )
    data_dict = {"year": pivot.index.tolist()}
    for org in top_orgs:
        data_dict[org] = pivot[org].tolist()
    org_trend_source.data = data_dict


def update_target_severity(subset: pd.DataFrame) -> None:
    if subset.empty:
        severity_source.data = dict(target=[], killed=[], wounded=[])
        target_severity_fig.x_range.factors = []
        return

    severity = (
        subset.groupby("targtype1_txt")
        .agg(killed=("nkill", "sum"), wounded=("nwound", "sum"))
        .reindex(top_target_types)
        .fillna(0)
    )
    severity_source.data = dict(
        target=severity.index.tolist(),
        killed=severity["killed"].tolist(),
        wounded=severity["wounded"].tolist(),
    )
    target_severity_fig.x_range.factors = severity.index.tolist()


def update_success_split(subset: pd.DataFrame) -> None:
    if subset.empty:
        success_source.data = dict(
            category=[],
            angle=[],
            color=[],
            label=[],
            incidents=[],
            percent=[],
            label_x=[],
            label_y=[],
            label_short=[],
        )
        return

    counts = subset["success_flag"].value_counts().reindex(["Successful", "Failed"]).fillna(0).astype(int)
    total = counts.sum()
    if total == 0:
        success_source.data = dict(
            category=[],
            angle=[],
            color=[],
            label=[],
            incidents=[],
            percent=[],
            label_x=[],
            label_y=[],
            label_short=[],
        )
        return

    angles = counts / total * 2 * math.pi
    palette = ["#2a9d8f", "#e76f51"]
    labels = [f"{cat}: {count:,} ({(count / total):.1%})" for cat, count in zip(counts.index, counts.values)]
    starts = np.append([0.0], np.cumsum(angles)[:-1])
    label_angles = starts + angles / 2
    label_radius = 0.45
    label_x = (np.cos(label_angles) * label_radius).tolist()
    label_y = (np.sin(label_angles) * label_radius).tolist()
    label_short = [f"{cat}\n{(count / total):.1%}" for cat, count in zip(counts.index, counts.values)]
    success_source.data = dict(
        category=counts.index.tolist(),
        angle=angles.tolist(),
        color=palette[: len(counts)],
        label=labels,
        incidents=counts.values.tolist(),
        percent=(counts / total * 100).tolist(),
        label_x=label_x,
        label_y=label_y,
        label_short=label_short,
    )


def update_table(subset: pd.DataFrame) -> None:
    if subset.empty:
        table_source.data = {key: [] for key in table_source.data.keys()}
        return
    top_events = (
        subset.sort_values("casualties", ascending=False)
        .head(25)[
            [
                "event_date",
                "country_txt",
                "city",
                "attacktype1_txt",
                "targtype1_txt",
                "nkill",
                "nwound",
                "casualties",
            ]
        ]
        .copy()
    )
    top_events["event_date"] = top_events["event_date"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    table_source.data = top_events.to_dict(orient="list")


def update_summary(subset: pd.DataFrame) -> None:
    incident_count = len(subset)
    fatalities = int(subset["nkill"].sum())
    wounded = int(subset["nwound"].sum())
    casualties = fatalities + wounded
    success_rate = subset["success_flag"].eq("Successful").mean() if incident_count else 0
    suicide_rate = subset["suicide_flag"].eq("Suicide").mean() if incident_count else 0
    avg_fatalities = subset["nkill"].mean() if incident_count else 0
    avg_wounded = subset["nwound"].mean() if incident_count else 0
    summary_div.text = f"""
        <div style='display:flex; gap:18px; flex-wrap:wrap;'>
            <div><strong>Incidents</strong><br>{incident_count:,}</div>
            <div><strong>Fatalities</strong><br>{fatalities:,}</div>
            <div><strong>Wounded</strong><br>{wounded:,}</div>
            <div><strong>Casualties</strong><br>{casualties:,}</div>
            <div><strong>Success rate</strong><br>{success_rate:0.1%}</div>
            <div><strong>Suicide share</strong><br>{suicide_rate:0.1%}</div>
            <div><strong>Avg killed/event</strong><br>{avg_fatalities:0.2f}</div>
            <div><strong>Avg wounded/event</strong><br>{avg_wounded:0.2f}</div>
        </div>
    """


def update_dashboard() -> None:
    subset = filter_data()
    update_map(subset)
    update_timeline(subset)
    update_country_bar(subset)
    update_attack_bar(subset)
    update_target_bar(subset)
    update_seasonality(subset)
    update_heatmap(subset)
    update_region_stack(subset)
    update_region_trend(subset)
    update_weapon_bar(subset)
    update_period_view(subset)
    update_hotspots(subset)
    update_attack_target_matrix(subset)
    update_weapon_share(subset)
    update_org_trend(subset)
    update_target_severity(subset)
    update_success_split(subset)
    update_table(subset)
    update_summary(subset)


def trigger_update(attr, old, new) -> None:
    update_dashboard()


def reset_filters() -> None:
    year_slider.value = (max(year_min, 1990), year_max)
    region_select.value = regions
    attack_select.value = attack_types
    target_select.value = target_types
    fatality_slider.value = (0, int(np.quantile(data["nkill"], 0.95)))
    casualty_slider.value = (0, int(np.quantile(data["casualties"], 0.95)))
    success_select.value = "All"
    suicide_select.value = "All"
    highlight_region_select.value = "None"
    if hotspot_regions:
        hotspot_region_select.value = hotspot_regions[0]
    update_dashboard()


controls = column(
    Div(text="<b>Global Filters</b>", styles={"font-size": "15px", "margin-bottom": "4px"}),
    year_slider,
    region_select,
    attack_select,
    target_select,
    fatality_slider,
    casualty_slider,
    success_select,
    suicide_select,
    timeline_metric,
    highlight_region_select,
    reset_button,
    width=330,
    sizing_mode="stretch_height",
    styles={**CARD_STYLE, "gap": "10px", "position": "sticky", "top": "10px"},
)

map_card = column(
    Div(text="<b>Global Spatial Distribution</b>", styles={"margin-bottom": "4px"}),
    map_fig,
    summary_div,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "10px"},
)

temporal_card = column(
    Div(text="<b>Time Evolution</b>", styles={"margin-bottom": "4px"}),
    timeline_fig,
    Div(text="<b>Monthly Seasonality</b>", styles={"margin-top": "10px"}),
    season_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "10px"},
)

country_card = column(
    Div(text="<b>High Casualty Countries</b>"),
    country_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

attack_card = column(
    Div(text="<b>Attack Type Composition</b>"),
    attack_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

target_card = column(
    Div(text="<b>Major Targets</b>"),
    target_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

region_card = column(
    Div(text="<b>Regional Casualty Trends</b>"),
    region_trend_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

weapon_card = column(
    Div(text="<b>Weapon Type Preference</b>"),
    weapon_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

success_card = column(
    Div(text="<b>Operation Outcome Share</b>"),
    success_pie_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px", "align-items": "center"},
)

period_card = column(
    Div(text="<b>Historical Period Comparison</b>"),
    period_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

hotspot_card = column(
    Div(text="<b>Regional Key Cities</b>"),
    hotspot_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

attack_target_card = column(
    Div(text="<b>Attack Type vs Target Matrix</b>"),
    attack_target_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

tableau_card = column(
    Div(text="<b>Tableau Embedded View</b>"),
    tableau_embed_div,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

weapon_share_card = column(
    Div(text="<b>Weapon Share Time Series</b>"),
    weapon_share_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

org_card = column(
    Div(text="<b>Major Organization Activity Trends</b>"),
    org_trend_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

severity_card = column(
    Div(text="<b>Target Casualty Structure</b>"),
    target_severity_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

heatmap_card = column(
    Div(text="<b>Decade × Region Heatmap</b>"),
    heatmap_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

table_card = column(
    Div(text="<b>Highest Casualty Events</b>"),
    table,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

region_stack_card = column(
    Div(text="<b>Regional Stacked Area</b>"),
    region_stack_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

tableau_section = column(tableau_card, sizing_mode="stretch_width", styles={"gap": "14px"})

top_section = row(
    controls,
    column(map_card, temporal_card, region_stack_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    sizing_mode="stretch_width",
    styles={"gap": "18px", "align-items": "flex-start"},
)

mid_section = row(
    country_card,
    attack_card,
    period_card,
    sizing_mode="stretch_width",
    styles={"gap": "14px"},
)

secondary_section = row(
    target_card,
    weapon_card,
    hotspot_card,
    sizing_mode="stretch_width",
    styles={"gap": "14px"},
)

extra_section = row(
    region_card,
    success_card,
    sizing_mode="stretch_width",
    styles={"gap": "14px"},
)

advanced_section = column(
    attack_target_card,
    weapon_share_card,
    org_card,
    severity_card,
    heatmap_card,
    table_card,
    sizing_mode="stretch_width",
    styles={"gap": "14px"},
)

header = Div(
    text="""
        <div style='display:flex; flex-direction:column; gap:4px;'>
            <h2 style='margin:0;'>Global Terrorism Intelligence Dashboard</h2>
            <p style='margin:0; color:#4b5563;'>
                Interactive multi-perspective dashboard visualizing global terrorism incidents (1970-2017) covering time, space, weapons, and targets.
            </p>
        </div>
    """,
    styles={**CARD_STYLE, "gap": "4px"},
)

dashboard = column(
    header,
    top_section,
    mid_section,
    secondary_section,
    extra_section,
    tableau_section,
    advanced_section,
    sizing_mode="stretch_width",
    styles={"gap": "16px", "background-color": "#f4f7fb", "padding": "16px"},
)
curdoc().add_root(dashboard)

curdoc().theme = "light_minimal"
curdoc().title = "Global Terrorism Intelligence Dashboard"


for widget in [
    year_slider,
    region_select,
    attack_select,
    target_select,
    fatality_slider,
    casualty_slider,
    success_select,
    suicide_select,
    timeline_metric,
    highlight_region_select,
    hotspot_region_select,
]:
    widget.on_change("value", trigger_update)

reset_button.on_click(reset_filters)

update_dashboard()

