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

import base64
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
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
    NumeralTickFormatter,
    Range1d,
    RangeSlider,
    Select,
    TableColumn,
    WMTSTileSource,
    GraphRenderer,
    StaticLayoutProvider,
    GraphRenderer,
    StaticLayoutProvider,
    MultiLine,
    Circle,
)
from bokeh.palettes import Turbo256, Category10, Category20
from bokeh.plotting import figure, from_networkx
from bokeh.transform import cumsum
import networkx as nx

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "globalterrorismdb_0718dist.zip"
BACKGROUND_IMAGE_PATH = BASE_PATH / "image.png"
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
REGION_ABBREV = {
    "Middle East & North Africa": "MENA",
    "South Asia": "SA",
    "Sub-Saharan Africa": "SSA",
    "South America": "SAM",
    "Central America & Caribbean": "CAC",
    "Southeast Asia": "SEA",
    "Western Europe": "WEU",
    "Eastern Europe": "EEU",
    "North America": "NAM",
    "East Asia": "EAS",
    "Central Asia": "CA",
    "Australasia & Oceania": "AUS",
}

def _smooth_curve(x0: float, y0: float, x1: float, y1: float, offset: float = 0.0) -> Tuple[List[float], List[float]]:
    steps = np.linspace(0, 1, 25)
    xs = x0 + (x1 - x0) * steps
    ys = y0 + (y1 - y0) * steps + offset * steps * (1 - steps)
    return xs.tolist(), ys.tolist()


def _load_bg_image(path: Path) -> str:
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        return ""


BG_IMAGE_DATA = _load_bg_image(BACKGROUND_IMAGE_PATH)
BG_STYLE = f"url('data:image/png;base64,{BG_IMAGE_DATA}')" if BG_IMAGE_DATA else "none"


def to_mercator(lat: pd.Series, lon: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Convert latitude/longitude to Web Mercator."""
    k = 6378137
    x = lon * (math.pi / 180) * k
    y = np.log(np.tan((90 + lat) * math.pi / 360)) * k
    return x, y


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load and enrich the dataset."""
    use_cols = [
        "iyear",
        "imonth",
        "iday",
        "eventid",
        "country_txt",
        "region_txt",
        "provstate",
        "city",
        "latitude",
        "longitude",
        "success",
        "suicide",
        "attacktype1_txt",
        "targtype1_txt",
        "gname",
        "weaptype1_txt",
        "nkill",
        "nwound",
    ]
    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False, usecols=use_cols)
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
top_attack_types = data["attacktype1_txt"].value_counts().head(6).index.tolist() or ["Unknown attack"]
region_palette = Category20[20]
REGION_COLOR_MAP = {region: region_palette[i % len(region_palette)] for i, region in enumerate(regions)}
ATTACK_COLOR_MAP = {attack: Category20[20][i % len(Category20[20])] for i, attack in enumerate(attack_types)}

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

region_share_source = ColumnDataSource(
    data=dict(decade=[], **{region: [] for region in regions})
)
attack_share_source = ColumnDataSource(
    data=dict(year=[], **{attack: [] for attack in top_attack_types})
)
attack_lethality_source = ColumnDataSource(
    data=dict(attack=[], incidents=[], casualties=[], avg=[], size=[], color=[])
)
target_percent_source = ColumnDataSource(data=dict(year=[], **{target: [] for target in top_target_types}))
sankey_edge_source = ColumnDataSource(
    data=dict(xs=[], ys=[], line_width=[], color=[], label=[], incidents=[])
)
sankey_node_source = ColumnDataSource(data=dict(x=[], y=[], name=[], color=[], type=[], value=[]))
org_network_source = ColumnDataSource(data=dict())
org_split_source = ColumnDataSource(data=dict(year=[], organized=[], unorganized=[]))
severity_dual_source = ColumnDataSource(data=dict(year=[], incidents=[], avg_casualties=[]))

boxplot_source = ColumnDataSource(
    data=dict(region=[], region_full=[], q1=[], q2=[], q3=[], lower=[], upper=[])
)
scatter_source = ColumnDataSource(
    data=dict(
        fatalities=[],
        wounded=[],
        casualties=[],
        country=[],
        city=[],
        year=[],
        attacktype=[],
        size=[],
    )
)
circle_source = ColumnDataSource(data=dict(category=[], incidents=[], casualties=[], size=[], color=[]))
attack_region_source = ColumnDataSource(
    data=dict(year=[], region=[], attack=[], incidents=[], color=[], alpha=[])
)
suicide_trend_source = ColumnDataSource(data=dict(year=[], rate=[]))
tactic_success_source = ColumnDataSource(data=dict(attack=[], rate=[], color=[]))


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
region_stack_renderers = region_stack_fig.varea_stack(
    stackers=regions,
    x="year",
    color=stack_colors,
    legend_label=regions,
    source=region_stack_source,
)
region_stack_hover = HoverTool(
    renderers=region_stack_renderers,
    tooltips=[
        ("Year", "$x{0}"),
        ("Region", "$name"),
        ("Incidents", "@$name{0,0}"),
    ],
    mode="vline",
)
region_stack_fig.add_tools(region_stack_hover)
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
    height=360,
    width=780,
    x_axis_label="Year",
    y_axis_label="Share of incidents",
    y_range=(0, 1),
    tools="pan,xwheel_zoom,reset,save",
)
weapon_share_renderers = weapon_share_fig.varea_stack(
    stackers=top_weapon_types,
    x="year",
    color=weapon_share_colors[: len(top_weapon_types)],
    legend_label=top_weapon_types,
    source=weapon_share_source,
)
for weapon, renderer in zip(top_weapon_types, weapon_share_renderers):
    renderer.name = weapon
weapon_share_fig.legend.location = "top_left"
weapon_share_fig.legend.click_policy = "hide"
for weapon, renderer in zip(top_weapon_types, weapon_share_renderers):
    weapon_share_fig.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("Year", "@year"),
                ("Weapon", weapon),
                ("Share", f"@{weapon}{{0.0%}}"),
            ],
            mode="vline",
        )
    )

boxplot_fig = figure(
    title="Regional fatalities distribution (boxplot)",
    height=360,
    width=600,
    sizing_mode="stretch_width",
    x_range=FactorRange(),
    toolbar_location=None,
)
boxplot_fig.segment(x0="region", y0="upper", x1="region", y1="q3", source=boxplot_source, line_width=2)
boxplot_fig.segment(x0="region", y0="lower", x1="region", y1="q1", source=boxplot_source, line_width=2)
boxplot_fig.vbar(
    x="region",
    width=0.7,
    top="q3",
    bottom="q1",
    source=boxplot_source,
    fill_color="#E76F51",
    fill_alpha=0.6,
    line_color="#333",
)
boxplot_fig.segment(
    x0="region",
    y0="lower",
    x1="region",
    y1="upper",
    source=boxplot_source,
    line_color="#333",
)
boxplot_fig.scatter(
    x="region",
    y="q2",
    size=8,
    color="#1D3557",
    source=boxplot_source,
)
box_hover = HoverTool(
    tooltips=[
        ("Region", "@region_full (@region)"),
        ("Q1", "@q1{0.0}"),
        ("Median", "@q2{0.0}"),
        ("Q3", "@q3{0.0}"),
        ("Lower whisker", "@lower{0.0}"),
        ("Upper whisker", "@upper{0.0}"),
    ]
)
boxplot_fig.add_tools(box_hover)

scatter_fig = figure(
    title="Event severity scatter (fatalities vs wounded)",
    height=360,
    width=600,
    sizing_mode="stretch_width",
    x_axis_label="Fatalities",
    y_axis_label="Wounded",
    tools="pan,wheel_zoom,reset,save",
)
scatter_renderer = scatter_fig.scatter(
    x="fatalities",
    y="wounded",
    size="size",
    source=scatter_source,
    fill_color="#43AA8B",
    fill_alpha=0.6,
    line_color="#1b4332",
)
scatter_hover = HoverTool(
    renderers=[scatter_renderer],
    tooltips=[
        ("Location", "@city, @country_txt"),
        ("Year", "@year"),
        ("Attack Type", "@attacktype"),
        ("Fatalities", "@fatalities{0,0}"),
        ("Wounded", "@wounded{0,0}"),
        ("Total casualties", "@casualties{0,0}"),
    ],
)
scatter_fig.add_tools(scatter_hover)

circle_fig = figure(
    title="Circle view: attack type impact",
    height=360,
    width=600,
    sizing_mode="stretch_width",
    toolbar_location=None,
)
circle_renderer = circle_fig.circle(
    x="incidents",
    y="casualties",
    size="size",
    source=circle_source,
    color="color",
    alpha=0.7,
    line_color="#1f2933",
)
circle_hover = HoverTool(
    renderers=[circle_renderer],
    tooltips=[
        ("Attack type", "@category"),
        ("Incidents", "@incidents{0,0}"),
        ("Casualties", "@casualties{0,0}"),
    ],
)
circle_fig.add_tools(circle_hover)
circle_fig.xaxis.axis_label = "Incidents"
circle_fig.yaxis.axis_label = "Casualties"

region_share_fig = figure(
    title="Regional share by decade",
    height=380,
    width=700,
    x_range=FactorRange(),
    y_range=(0, 1),
    toolbar_location=None,
)
region_share_renderers = region_share_fig.vbar_stack(
    stackers=regions,
    x="decade",
    width=0.8,
    color=[REGION_COLOR_MAP[r] for r in regions],
    legend_label=regions,
    source=region_share_source,
)
region_share_fig.yaxis.formatter = NumeralTickFormatter(format="0%")
region_share_fig.legend.click_policy = "mute"
region_share_hover = HoverTool(
    renderers=region_share_renderers,
    tooltips=[
        ("Decade", "@decade"),
        ("Region", "$name"),
        ("Share", "@$name{0.0%}"),
    ],
)
region_share_fig.add_tools(region_share_hover)

attack_share_fig = figure(
    title="Attack type share over time",
    height=360,
    width=780,
    x_axis_label="Year",
    y_axis_label="Share of incidents",
    tools="pan,xwheel_zoom,reset,save",
)
attack_share_renderers = attack_share_fig.varea_stack(
    stackers=top_attack_types,
    x="year",
    color=[ATTACK_COLOR_MAP[t] for t in top_attack_types],
    legend_label=top_attack_types,
    source=attack_share_source,
)
attack_share_fig.legend.location = "top_left"
attack_share_fig.legend.click_policy = "hide"
attack_share_hover = HoverTool(
    renderers=attack_share_renderers,
    tooltips=[("Year", "$x{0}"), ("Attack", "$name"), ("Share", "@$name{0.0%}")],
    mode="vline",
)
attack_share_fig.add_tools(attack_share_hover)

attack_lethality_fig = figure(
    title="Attack method lethality",
    height=360,
    width=600,
    x_axis_label="Incidents",
    y_axis_label="Casualties",
    tools="pan,wheel_zoom,reset,save",
)
attack_lethality_renderer = attack_lethality_fig.scatter(
    x="incidents",
    y="casualties",
    size="size",
    color="color",
    alpha=0.7,
    source=attack_lethality_source,
)
attack_lethality_hover = HoverTool(
    renderers=[attack_lethality_renderer],
    tooltips=[
        ("Attack type", "@attack"),
        ("Incidents", "@incidents{0,0}"),
        ("Casualties", "@casualties{0,0}"),
        ("Avg casualties/event", "@avg{0.0}"),
    ],
)
attack_lethality_fig.add_tools(attack_lethality_hover)

target_percent_fig = figure(
    title="Target distribution over time",
    height=360,
    width=780,
    x_axis_label="Year",
    y_axis_label="Share",
    tools="pan,xwheel_zoom,reset,save",
)
target_percent_renderers = target_percent_fig.varea_stack(
    stackers=top_target_types,
    x="year",
    color=[Category20[20][i % 20] for i in range(len(top_target_types))],
    legend_label=top_target_types,
    source=target_percent_source,
)
target_percent_fig.legend.location = "top_left"
target_percent_fig.legend.click_policy = "hide"
target_percent_hover = HoverTool(
    renderers=target_percent_renderers,
    tooltips=[("Year", "$x{0}"), ("Target", "$name"), ("Share", "@$name{0.0%}")],
    mode="vline",
)
target_percent_fig.add_tools(target_percent_hover)

sankey_fig = figure(
    title="Attack → Target → Outcome flow",
    height=420,
    width=780,
    x_range=(-0.2, 2.2),
    y_range=(-0.5, 12.5),
    toolbar_location=None,
)
sankey_edge_renderer = sankey_fig.multi_line(
    xs="xs",
    ys="ys",
    line_width="line_width",
    color="color",
    alpha=0.6,
    source=sankey_edge_source,
)
sankey_node_renderer = sankey_fig.circle(
    x="x",
    y="y",
    size=18,
    color="color",
    alpha=0.9,
    source=sankey_node_source,
)
sankey_node_hover = HoverTool(
    renderers=[sankey_node_renderer],
    tooltips=[("Node", "@name"), ("Type", "@type"), ("Value", "@value{0,0}")],
)
sankey_edge_hover = HoverTool(
    renderers=[sankey_edge_renderer],
    tooltips=[("Flow", "@label"), ("Incidents", "@incidents{0,0}")],
)
sankey_fig.add_tools(sankey_node_hover, sankey_edge_hover)

org_network_fig = figure(
    title="Organization relationship network",
    height=520,
    width=720,
    x_range=(-2.0, 2.0),
    y_range=(-2.0, 2.0),
    toolbar_location=None,
)
org_graph = GraphRenderer()
org_graph.node_renderer.glyph = Circle(radius=0.05, fill_color="color")
org_graph.edge_renderer.glyph = MultiLine(line_color="line_color", line_alpha=0.6, line_width="line_width")
org_graph.layout_provider = StaticLayoutProvider(graph_layout={})
org_network_fig.renderers.append(org_graph)
org_network_node_hover = HoverTool(
    renderers=[org_graph.node_renderer],
    tooltips=[("Node", "@label"), ("Type", "@type"), ("Incidents", "@value{0,0}")],
)
org_network_fig.add_tools(org_network_node_hover)

org_split_fig = figure(
    title="Organized vs unaffiliated trend",
    height=360,
    width=780,
    x_axis_label="Year",
    y_axis_label="Incidents",
    tools="pan,xwheel_zoom,reset,save",
)
organized_line = org_split_fig.line(
    x="year",
    y="organized",
    color="#1d3557",
    line_width=3,
    source=org_split_source,
    legend_label="Organized",
)
unorganized_line = org_split_fig.line(
    x="year",
    y="unorganized",
    color="#e76f51",
    line_width=3,
    source=org_split_source,
    legend_label="Unaffiliated",
)
org_split_fig.legend.location = "top_left"
org_split_hover = HoverTool(
    renderers=[organized_line, unorganized_line],
    tooltips=[("Year", "@year"), ("Incidents", "$y{0,0}")],
    mode="vline",
)
org_split_fig.add_tools(org_split_hover)

severity_dual_fig = figure(
    title="Incident vs average casualties",
    height=360,
    width=780,
    x_axis_label="Year",
    y_axis_label="Incidents",
    tools="pan,xwheel_zoom,reset,save",
)
severity_dual_fig.extra_y_ranges = {"avg": Range1d(start=0, end=5)}
severity_dual_fig.add_layout(LinearAxis(y_range_name="avg", axis_label="Avg casualties per event"), "right")
severity_incidents = severity_dual_fig.line(
    x="year",
    y="incidents",
    color="#2196f3",
    line_width=3,
    source=severity_dual_source,
)
severity_avg = severity_dual_fig.line(
    x="year",
    y="avg_casualties",
    color="#f44336",
    line_width=3,
    y_range_name="avg",
    source=severity_dual_source,
)
severity_hover = HoverTool(
    renderers=[severity_incidents, severity_avg],
    tooltips=[
        ("Year", "@year"),
        ("Incidents", "@incidents{0,0}"),
        ("Avg casualty/event", "@avg_casualties{0.00}"),
    ],
    mode="vline",
)
severity_dual_fig.add_tools(severity_hover)

attack_region_fig = figure(
    title="Dominant attack type by region and year",
    height=360,
    width=780,
    sizing_mode="stretch_width",
    x_axis_label="Year",
    y_range=FactorRange(),
    tools="pan,xwheel_zoom,reset,save",
)
attack_region_renderer = attack_region_fig.rect(
    x="year",
    y="region",
    width=1,
    height=0.9,
    source=attack_region_source,
    fill_color="color",
    fill_alpha="alpha",
    line_color=None,
)
attack_region_hover = HoverTool(
    renderers=[attack_region_renderer],
    tooltips=[
        ("Region", "@region"),
        ("Year", "@year"),
        ("Dominant attack", "@attack"),
        ("Incidents", "@incidents{0,0}"),
    ],
)
attack_region_fig.add_tools(attack_region_hover)

suicide_trend_fig = figure(
    title="Suicide attack rate trend",
    height=360,
    width=600,
    sizing_mode="stretch_width",
    x_axis_label="Year",
    y_axis_label="Suicide share",
    tools="pan,xwheel_zoom,reset,save",
)
suicide_line = suicide_trend_fig.line(
    x="year",
    y="rate",
    source=suicide_trend_source,
    line_width=3,
    color="#ef476f",
)
suicide_scatter = suicide_trend_fig.scatter(
    x="year",
    y="rate",
    size=7,
    color="#ef476f",
    source=suicide_trend_source,
)
suicide_hover = HoverTool(
    renderers=[suicide_scatter],
    tooltips=[("Year", "@year"), ("Suicide rate", "@rate{0.0%}")],
    mode="vline",
)
suicide_trend_fig.add_tools(suicide_hover)

tactic_success_fig = figure(
    title="Success rate by attack tactic",
    height=360,
    width=600,
    sizing_mode="stretch_width",
    y_range=FactorRange(),
    x_axis_label="Success rate",
    toolbar_location=None,
)
tactic_renderer = tactic_success_fig.hbar(
    y="attack",
    right="rate",
    height=0.6,
    source=tactic_success_source,
    color="color",
)
tactic_hover = HoverTool(
    renderers=[tactic_renderer],
    tooltips=[("Attack type", "@attack"), ("Success rate", "@rate{0.0%}")],
)
tactic_success_fig.add_tools(tactic_hover)

org_colors = Category20[20]
org_trend_fig = figure(
    title="Top organizations incident volume",
    height=320,
    width=780,
    x_axis_label="Year",
    y_axis_label="Incidents",
    tools="pan,xwheel_zoom,reset,save",
)
org_renderers = org_trend_fig.varea_stack(
    stackers=top_orgs,
    x="year",
    color=org_colors[: len(top_orgs)],
    legend_label=top_orgs,
    source=org_trend_source,
)
for org_name, renderer in zip(top_orgs, org_renderers):
    renderer.name = org_name
org_trend_fig.legend.location = "top_left"
org_trend_fig.legend.click_policy = "hide"
for org_name, renderer in zip(top_orgs, org_renderers):
    org_trend_fig.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("Year", "@year"),
                ("Organization", org_name),
                ("Incidents", f"@{org_name}{{0,0}}"),
            ],
            mode="vline",
        )
    )

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


def update_region_share(subset: pd.DataFrame) -> None:
    if subset.empty:
        region_share_source.data = {"decade": []}
        return
    share_df = subset.copy()
    share_df["decade"] = (share_df["iyear"] // 10) * 10
    share_df = share_df.dropna(subset=["decade"])
    decade_order = sorted(share_df["decade"].unique())
    data = {"decade": [str(int(d)) for d in decade_order]}
    grouped = share_df.groupby(["decade", "region_txt"]).size().rename("incidents").reset_index()
    totals = grouped.groupby("decade")["incidents"].transform("sum")
    grouped["share"] = grouped["incidents"] / totals
    for region in regions:
        values = []
        for d in decade_order:
            row = grouped[(grouped["decade"] == d) & (grouped["region_txt"] == region)]
            values.append(row["share"].iloc[0] if not row.empty else 0.0)
        data[region] = values
    region_share_source.data = data
    region_share_fig.x_range.factors = data["decade"]


def update_attack_share(subset: pd.DataFrame) -> None:
    if subset.empty:
        attack_share_source.data = {"year": []}
        return
    share = (
        subset[subset["attacktype1_txt"].isin(top_attack_types)]
        .groupby(["iyear", "attacktype1_txt"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_attack_types, fill_value=0)
        .sort_index()
    )
    totals = share.sum(axis=1).replace(0, 1)
    share = share.div(totals, axis=0)
    data = {"year": share.index.tolist()}
    for attack in top_attack_types:
        data[attack] = share[attack].tolist()
    attack_share_source.data = data


def update_attack_lethality(subset: pd.DataFrame) -> None:
    if subset.empty:
        attack_lethality_source.data = dict(attack=[], incidents=[], casualties=[], avg=[], size=[], color=[])
        return
    agg = (
        subset.groupby("attacktype1_txt")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .reset_index()
        .rename(columns={"attacktype1_txt": "attack"})
    )
    agg["avg"] = agg["casualties"] / agg["incidents"]
    max_avg = max(agg["avg"].max(), 1)
    agg["size"] = np.interp(agg["avg"], [0, max_avg], [10, 50])
    agg["color"] = agg["attack"].map(ATTACK_COLOR_MAP).fillna("#888888")
    attack_lethality_source.data = agg.to_dict(orient="list")


def update_target_percent(subset: pd.DataFrame) -> None:
    if subset.empty:
        target_percent_source.data = {"year": []}
        return
    subset = subset[subset["targtype1_txt"].isin(top_target_types)]
    share = (
        subset.groupby(["iyear", "targtype1_txt"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_target_types, fill_value=0)
        .sort_index()
    )
    totals = share.sum(axis=1).replace(0, 1)
    share = share.div(totals, axis=0)
    data = {"year": share.index.tolist()}
    for target in top_target_types:
        data[target] = share[target].tolist()
    target_percent_source.data = data


def update_sankey(subset: pd.DataFrame) -> None:
    if subset.empty:
        sankey_edge_source.data = dict(xs=[], ys=[], line_width=[], color=[], label=[], incidents=[])
        sankey_node_source.data = dict(x=[], y=[], name=[], color=[], type=[], value=[])
        return
    top_attacks = subset["attacktype1_txt"].value_counts().head(5).index.tolist()
    top_targets = subset["targtype1_txt"].value_counts().head(5).index.tolist()
    outcomes = ["Successful", "Failed"]
    gap = 1.8
    def _positions(count: int) -> np.ndarray:
        if count <= 1:
            return np.array([0.0])
        return np.linspace(0, (count - 1) * gap, count)
    attack_nodes = list(zip([0.0] * len(top_attacks), _positions(len(top_attacks))))
    target_nodes = list(zip([1.0] * len(top_targets), _positions(len(top_targets))))
    outcome_nodes = list(zip([2.0] * len(outcomes), _positions(len(outcomes))))
    all_y = [y for _, y in attack_nodes + target_nodes + outcome_nodes]
    if all_y:
        sankey_fig.y_range.start = min(all_y) - gap
        sankey_fig.y_range.end = max(all_y) + gap
    else:
        sankey_fig.y_range.start = -gap
        sankey_fig.y_range.end = gap
    node_x = []
    node_y = []
    names = []
    colors = []
    palette = Category20[20]
    for idx, (x, y) in enumerate(attack_nodes):
        node_x.append(x)
        node_y.append(y)
        names.append(top_attacks[idx])
        colors.append(palette[idx % 20])
    for idx, (x, y) in enumerate(target_nodes):
        node_x.append(x)
        node_y.append(y)
        names.append(top_targets[idx])
        colors.append(palette[(idx + 5) % 20])
    for idx, (x, y) in enumerate(outcome_nodes):
        node_x.append(x)
        node_y.append(y)
        names.append(outcomes[idx])
        colors.append(palette[(idx + 10) % 20])
    values = []
    types = []
    attack_totals = (
        subset[subset["attacktype1_txt"].isin(top_attacks)]
        .groupby("attacktype1_txt")["eventid"]
        .count()
        .to_dict()
    )
    target_totals = (
        subset[subset["targtype1_txt"].isin(top_targets)]
        .groupby("targtype1_txt")["eventid"]
        .count()
        .to_dict()
    )
    outcome_totals = (
        subset.groupby("success_flag")["eventid"].count().to_dict()
    )
    for idx in range(len(attack_nodes)):
        attack = top_attacks[idx]
        values.append(attack_totals.get(attack, 0))
        types.append("Attack")
    for idx in range(len(target_nodes)):
        target = top_targets[idx]
        values.append(target_totals.get(target, 0))
        types.append("Target")
    for idx in range(len(outcome_nodes)):
        outcome = outcomes[idx]
        values.append(outcome_totals.get(outcome, 0))
        types.append("Outcome")
    sankey_node_source.data = dict(x=node_x, y=node_y, name=names, color=colors, type=types, value=values)
    xs = []
    ys = []
    widths = []
    edge_colors = []
    labels = []
    incidents_list = []
    attack_target = (
        subset[subset["attacktype1_txt"].isin(top_attacks)]
        .groupby(["attacktype1_txt", "targtype1_txt"])
        .size()
        .reset_index(name="incidents")
    )
    max_val = max(attack_target["incidents"].max(), 1) if not attack_target.empty else 1
    for _, row in attack_target.iterrows():
        if row["targtype1_txt"] not in top_targets:
            continue
        i_attack = top_attacks.index(row["attacktype1_txt"])
        i_target = top_targets.index(row["targtype1_txt"])
        offset = (i_target - len(top_targets) / 2) * 0.35
        curve_x, curve_y = _smooth_curve(attack_nodes[i_attack][0], attack_nodes[i_attack][1], target_nodes[i_target][0], target_nodes[i_target][1], offset)
        xs.append(curve_x)
        ys.append(curve_y)
        widths.append(2 + 10 * (row["incidents"] / max_val))
        edge_colors.append(palette[i_attack % 20])
        labels.append(f"{row['attacktype1_txt']} → {row['targtype1_txt']}")
        incidents_list.append(row["incidents"])
    target_outcome = (
        subset[subset["targtype1_txt"].isin(top_targets)]
        .groupby(["targtype1_txt", "success_flag"])
        .size()
        .reset_index(name="incidents")
    )
    max_val = max(target_outcome["incidents"].max(), 1) if not target_outcome.empty else 1
    for _, row in target_outcome.iterrows():
        if row["success_flag"] not in outcomes:
            continue
        if row["targtype1_txt"] not in top_targets:
            continue
        i_target = top_targets.index(row["targtype1_txt"])
        i_outcome = outcomes.index(row["success_flag"])
        offset = (i_outcome - len(outcomes) / 2) * 0.45
        curve_x, curve_y = _smooth_curve(target_nodes[i_target][0], target_nodes[i_target][1], outcome_nodes[i_outcome][0], outcome_nodes[i_outcome][1], offset)
        xs.append(curve_x)
        ys.append(curve_y)
        widths.append(2 + 10 * (row["incidents"] / max_val))
        edge_colors.append(palette[(i_outcome + 10) % 20])
        labels.append(f"{row['targtype1_txt']} → {row['success_flag']}")
        incidents_list.append(row["incidents"])
    sankey_edge_source.data = dict(
        xs=xs,
        ys=ys,
        line_width=widths,
        color=edge_colors,
        label=labels,
        incidents=incidents_list,
    )


def update_org_network(subset: pd.DataFrame) -> None:
    filtered = subset[subset["gname"].isin(top_orgs) & subset["region_txt"].notna()]
    if filtered.empty:
        org_graph.node_renderer.data_source.data = dict(index=[], color=[], label=[], type=[], value=[])
        org_graph.edge_renderer.data_source.data = dict(start=[], end=[], line_width=[], line_color=[])
        org_graph.layout_provider.graph_layout = {}
        return
    G = nx.Graph()
    for org in top_orgs:
        G.add_node(f"ORG:{org}", color="#1d3557")
    region_nodes = sorted(filtered["region_txt"].unique().tolist())
    for region in region_nodes:
        G.add_node(f"REG:{region}", color="#e76f51")
    edges = (
        filtered.groupby(["gname", "region_txt"])
        .size()
        .reset_index(name="incidents")
    )
    max_val = max(edges["incidents"].max(), 1)
    for _, row in edges.iterrows():
        G.add_edge(
            f"ORG:{row['gname']}",
            f"REG:{row['region_txt']}",
            weight=2 + 8 * (row["incidents"] / max_val),
        )
    layout = nx.spring_layout(G, seed=42, k=0.7, scale=1.8)
    org_totals = filtered.groupby("gname")["eventid"].count().to_dict()
    region_totals = filtered.groupby("region_txt")["eventid"].count().to_dict()
    node_labels = []
    node_types = []
    node_colors = []
    node_values = []
    for node in G.nodes():
        if node.startswith("ORG:"):
            node_labels.append(node.replace("ORG:", ""))
            node_types.append("Organization")
            node_values.append(org_totals.get(node.replace("ORG:", ""), 0))
        else:
            node_labels.append(node.replace("REG:", ""))
            node_types.append("Region")
            node_values.append(region_totals.get(node.replace("REG:", ""), 0))
        node_colors.append(G.nodes[node]["color"])
    org_graph.node_renderer.data_source.data = dict(
        index=list(G.nodes()),
        color=node_colors,
        label=node_labels,
        type=node_types,
        value=node_values,
    )
    org_graph.edge_renderer.data_source.data = dict(
        start=[edge[0] for edge in G.edges()],
        end=[edge[1] for edge in G.edges()],
        line_width=[G.edges[edge]["weight"] for edge in G.edges()],
        line_color=["#94a3b8"] * len(G.edges()),
    )
    org_graph.layout_provider.graph_layout = layout


def update_org_split(subset: pd.DataFrame) -> None:
    if subset.empty:
        org_split_source.data = dict(year=[], organized=[], unorganized=[])
        return
    grouped = subset.groupby("iyear").agg(
        organized=("gname", lambda s: s.ne("Unknown group").sum()),
        total=("eventid", "count"),
    )
    grouped["unorganized"] = grouped["total"] - grouped["organized"]
    grouped = grouped[["organized", "unorganized"]].reset_index().rename(columns={"iyear": "year"})
    org_split_source.data = grouped.to_dict(orient="list")


def update_severity_dual(subset: pd.DataFrame) -> None:
    if subset.empty:
        severity_dual_source.data = dict(year=[], incidents=[], avg_casualties=[])
        return
    grouped = (
        subset.groupby("iyear")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .reset_index()
        .rename(columns={"iyear": "year"})
    )
    grouped["avg_casualties"] = grouped["casualties"] / grouped["incidents"]
    severity_dual_source.data = grouped[["year", "incidents", "avg_casualties"]].to_dict(orient="list")
    if not grouped.empty:
        severity_dual_fig.extra_y_ranges["avg"].start = 0
        severity_dual_fig.extra_y_ranges["avg"].end = float(grouped["avg_casualties"].max() * 1.2)

def update_boxplot(subset: pd.DataFrame) -> None:
    if subset.empty:
        boxplot_source.data = {key: [] for key in boxplot_source.data}
        boxplot_fig.x_range.factors = []
        return
    stats = []
    for region, values in subset.groupby("region_txt")["nkill"]:
        clean = values.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            continue
        q1, median, q3 = clean.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = max(clean.min(), q1 - 1.5 * iqr)
        upper = min(clean.max(), q3 + 1.5 * iqr)
        abbrev = REGION_ABBREV.get(region, "".join(word[0].upper() for word in region.split()))
        stats.append(
            dict(
                region=abbrev,
                region_full=region,
                q1=q1,
                q2=median,
                q3=q3,
                lower=lower,
                upper=upper,
            )
        )
    if not stats:
        boxplot_source.data = {key: [] for key in boxplot_source.data}
        boxplot_fig.x_range.factors = []
        return
    stats_df = pd.DataFrame(stats).sort_values("region_full")
    boxplot_source.data = stats_df.to_dict(orient="list")
    boxplot_fig.x_range.factors = stats_df["region"].tolist()


def update_scatter(subset: pd.DataFrame) -> None:
    if subset.empty:
        scatter_source.data = {key: [] for key in scatter_source.data}
        return
    scatter_df = (
        subset.sort_values("casualties", ascending=False)
        .head(500)[
            ["nkill", "nwound", "casualties", "country_txt", "city", "iyear", "attacktype1_txt"]
        ]
        .rename(
            columns={
                "nkill": "fatalities",
                "nwound": "wounded",
                "iyear": "year",
                "attacktype1_txt": "attacktype",
            }
        )
    )
    min_c, max_c = scatter_df["casualties"].min(), scatter_df["casualties"].max()
    scatter_df["casualties"] = scatter_df["casualties"].astype(float)
    if max_c == min_c:
        scatter_df["size"] = 15
    else:
        scatter_df["size"] = np.interp(scatter_df["casualties"], [min_c, max_c], [8, 35])
    scatter_source.data = scatter_df.to_dict(orient="list")


def update_circle_view(subset: pd.DataFrame) -> None:
    if subset.empty:
        circle_source.data = {key: [] for key in circle_source.data}
        return
    attack_summary = (
        subset.groupby("attacktype1_txt")
        .agg(incidents=("eventid", "count"), casualties=("casualties", "sum"))
        .sort_values("incidents", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"attacktype1_txt": "category"})
    )
    max_casualty = max(attack_summary["casualties"].max(), 1)
    attack_summary["size"] = np.interp(attack_summary["casualties"], [0, max_casualty], [15, 60])
    attack_summary["color"] = attack_summary["category"].map(ATTACK_COLOR_MAP)
    circle_source.data = attack_summary.to_dict(orient="list")


def update_attack_region(subset: pd.DataFrame) -> None:
    if subset.empty:
        attack_region_source.data = {key: [] for key in attack_region_source.data}
        attack_region_fig.y_range.factors = []
        return
    agg = (
        subset.groupby(["iyear", "region_txt", "attacktype1_txt"])
        .agg(incidents=("eventid", "count"))
        .reset_index()
    )
    dominant = (
        agg.sort_values("incidents", ascending=False)
        .groupby(["iyear", "region_txt"])
        .head(1)
        .reset_index(drop=True)
    )
    dominant["color"] = dominant["attacktype1_txt"].map(ATTACK_COLOR_MAP)
    min_inc, max_inc = dominant["incidents"].min(), dominant["incidents"].max()
    if max_inc == min_inc:
        dominant["alpha"] = 0.8
    else:
        dominant["alpha"] = np.interp(dominant["incidents"], [min_inc, max_inc], [0.4, 0.95])
    attack_region_source.data = dominant.rename(
        columns={"iyear": "year", "region_txt": "region", "attacktype1_txt": "attack"}
    ).to_dict(orient="list")
    attack_region_fig.y_range.factors = sorted(dominant["region_txt"].unique().tolist())
    attack_region_fig.x_range.start = dominant["iyear"].min() - 1
    attack_region_fig.x_range.end = dominant["iyear"].max() + 1


def update_suicide_trend(subset: pd.DataFrame) -> None:
    if subset.empty:
        suicide_trend_source.data = {key: [] for key in suicide_trend_source.data}
        return
    yearly = (
        subset.groupby("iyear")
        .agg(
            incidents=("eventid", "count"),
            suicide_incidents=("suicide_flag", lambda s: (s == "Suicide").sum()),
        )
        .reset_index()
        .rename(columns={"iyear": "year"})
    )
    yearly["rate"] = yearly["suicide_incidents"] / yearly["incidents"]
    suicide_trend_source.data = yearly[["year", "rate"]].to_dict(orient="list")
    suicide_trend_fig.x_range.start = yearly["year"].min()
    suicide_trend_fig.x_range.end = yearly["year"].max()


def update_tactic_success(subset: pd.DataFrame) -> None:
    if subset.empty:
        tactic_success_source.data = {key: [] for key in tactic_success_source.data}
        tactic_success_fig.y_range.factors = []
        return
    success = (
        subset.groupby("attacktype1_txt")
        .agg(rate=("success", "mean"))
        .reset_index()
        .sort_values("rate", ascending=False)
    )
    success["color"] = success["attacktype1_txt"].map(ATTACK_COLOR_MAP)
    tactic_success_source.data = success.rename(columns={"attacktype1_txt": "attack"}).to_dict(orient="list")
    tactic_success_fig.y_range.factors = success["attacktype1_txt"].tolist()


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
    metrics = [
        ("Incidents", f"{incident_count:,}"),
        ("Fatalities", f"{fatalities:,}"),
        ("Wounded", f"{wounded:,}"),
        ("Casualties", f"{casualties:,}"),
        ("Success rate", f"{success_rate:0.1%}"),
        ("Suicide share", f"{suicide_rate:0.1%}"),
        ("Avg killed/event", f"{avg_fatalities:0.2f}"),
        ("Avg wounded/event", f"{avg_wounded:0.2f}"),
    ]
    cards_html = "".join(
        [
            f"""
            <div class='summary-card'>
                <div class='summary-label'>{label}</div>
                <div class='summary-value'>{value}</div>
            </div>
            """
            for label, value in metrics
        ]
    )
    summary_div.text = f"""
        <style>
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(8, minmax(120px, 1fr));
                gap: 12px;
                width: 100%;
            }}
            .summary-card {{
                background-color: rgba(10, 14, 23, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 10px;
                padding: 12px 16px;
                color: #f1f5f9;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
            }}
            .summary-label {{
                font-size: 13px;
                letter-spacing: 0.3px;
                color: #e2e8f0;
                text-transform: uppercase;
            }}
            .summary-value {{
                font-size: 24px;
                font-weight: 600;
                margin-top: 4px;
            }}
        </style>
        <div class='summary-grid'>
            {cards_html}
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
    update_boxplot(subset)
    update_scatter(subset)
    update_circle_view(subset)
    update_attack_region(subset)
    update_suicide_trend(subset)
    update_tactic_success(subset)
    update_region_share(subset)
    update_attack_share(subset)
    update_attack_lethality(subset)
    update_target_percent(subset)
    update_sankey(subset)
    update_org_network(subset)
    update_org_split(subset)
    update_severity_dual(subset)
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
    styles={**CARD_STYLE, "gap": "6px", "height": "100%"},
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
    styles={**CARD_STYLE, "gap": "6px", "height": "100%"},
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
    styles={**CARD_STYLE, "gap": "6px", "height": "100%"},
)

table_card = column(
    Div(text="<b>Highest Casualty Events</b>"),
    table,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px", "height": "100%"},
)

region_stack_card = column(
    Div(text="<b>Regional Stacked Area</b>"),
    region_stack_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

boxplot_card = column(
    Div(text="<b>Fatalities distribution by region</b>"),
    boxplot_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

scatter_card = column(
    Div(text="<b>Event severity scatter</b>"),
    scatter_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

circle_card = column(
    Div(text="<b>Circle view: attack impact</b>"),
    circle_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

attack_region_card = column(
    Div(text="<b>Dominant attack types by region & year</b>"),
    attack_region_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

region_share_card = column(
    Div(text="<b>Regional share by decade</b>"),
    region_share_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

attack_share_card = column(
    Div(text="<b>Attack type share over time</b>"),
    attack_share_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

attack_lethality_card = column(
    Div(text="<b>Attack method lethality</b>"),
    attack_lethality_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

target_percent_card = column(
    Div(text="<b>Target distribution over time</b>"),
    target_percent_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

sankey_card = column(
    Div(text="<b>Attack → Target → Outcome flow</b>"),
    sankey_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

org_network_card = column(
    Div(text="<b>Organization relationship network</b>"),
    org_network_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

org_split_card = column(
    Div(text="<b>Organized vs unaffiliated trend</b>"),
    org_split_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

severity_dual_card = column(
    Div(text="<b>Incident vs average casualties</b>"),
    severity_dual_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

suicide_card = column(
    Div(text="<b>Deep insight · Suicide trend</b>"),
    suicide_trend_fig,
    sizing_mode="stretch_width",
    styles={**CARD_STYLE, "gap": "6px"},
)

tactic_card = column(
    Div(text="<b>Deep insight · Tactic success</b>"),
    tactic_success_fig,
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

deep_insight_section = row(
    suicide_card,
    tactic_card,
    sizing_mode="stretch_width",
    styles={"gap": "14px"},
)

post_tableau_section = column(
    row(region_share_card, attack_share_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(attack_lethality_card, target_percent_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(sankey_card, org_network_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(org_split_card, severity_dual_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(boxplot_card, scatter_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(circle_card, attack_region_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(weapon_share_card, attack_target_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(org_card, severity_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    row(heatmap_card, table_card, sizing_mode="stretch_width", styles={"gap": "14px"}),
    sizing_mode="stretch_width",
    styles={"gap": "16px"},
)

header = Div(
    text="""
        <div style='display:flex; flex-direction:column; gap:4px; margin-bottom:10px; color:#f8fafc;'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
                <h2 style='margin:0; font-weight:600;'>Global Terrorism Intelligence Dashboard</h2>
                <p style='margin:0; color:#cbd5f5;'>
                    Interactive multi-perspective dashboard visualizing global terrorism incidents (1970-2017).
                </p>
            </div>
        </div>
    """,
    sizing_mode="stretch_width",
    styles={
        **CARD_STYLE,
        "gap": "4px",
        "padding-bottom": "2px",
        "background-color": "rgba(15,17,25,0.85)",
        "border": "1px solid rgba(255,255,255,0.15)",
    },
)

summary_card = column(
    summary_div,
    sizing_mode="stretch_width",
    styles={
        "padding": "0",
        "background-color": "transparent",
        "box-shadow": "none",
        "border": "none",
        "width": "100%",
    },
)

dashboard = column(
    header,
    summary_card,
    top_section,
    mid_section,
    secondary_section,
    extra_section,
    deep_insight_section,
    tableau_section,
    post_tableau_section,
    sizing_mode="stretch_width",
    styles={
        "gap": "12px",
        "background-color": "#010409",
        "background-image": BG_STYLE,
        "background-size": "contain",
        "background-position": "center top",
        "background-repeat": "no-repeat",
        "padding": "16px",
    },
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

