"""
AgriCascade -- World-Class VC Pitch Deck Generator
Author: Antigravity (Google DeepMind)
Target: D:\\legion\\Roma Ali pitch-deck.pptx

Design rules:
 - NO 3-card container traps on any single slide
 - NO generic icons (lightbulbs, rockets, targets)
 - Unique structural layout per slide
 - Mature deep-space + agriculture color palette
 - Executive-level, Sequoia / YC presentation quality
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# ─────────────────────────────────────────────
#  GLOBAL DESIGN SYSTEM
# ─────────────────────────────────────────────
W = Inches(13.333)
H = Inches(7.5)

# Palette
BG_DARK      = RGBColor(0x0B, 0x13, 0x20)
BG_MID       = RGBColor(0x0F, 0x1E, 0x35)
PANEL_DARK   = RGBColor(0x11, 0x1E, 0x35)
PANEL_MID    = RGBColor(0x16, 0x28, 0x47)
PANEL_LIGHT  = RGBColor(0x1B, 0x33, 0x58)
BORDER_SUB   = RGBColor(0x22, 0x3A, 0x5E)

GREEN_PRI    = RGBColor(0x10, 0xB9, 0x81)
GREEN_DEEP   = RGBColor(0x05, 0x78, 0x52)
GREEN_PALE   = RGBColor(0x06, 0xD6, 0xA0)
CYAN_PRI     = RGBColor(0x0E, 0xA5, 0xE9)
AMBER        = RGBColor(0xF5, 0x9E, 0x0B)
AMBER_PALE   = RGBColor(0xFB, 0xBF, 0x24)
RED_ALERT    = RGBColor(0xF4, 0x3F, 0x5E)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
SLATE_100    = RGBColor(0xF1, 0xF5, 0xF9)
SLATE_300    = RGBColor(0xCB, 0xD5, 0xE1)
SLATE_400    = RGBColor(0x94, 0xA3, 0xB8)
SLATE_600    = RGBColor(0x47, 0x55, 0x69)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ─── Shape helpers ───────────────────────────

def fill_bg(slide, color=BG_DARK):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def rect(slide, left, top, width, height, fill=PANEL_DARK, border=None, border_pt=1.0, corner=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if corner else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_type, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if border:
        sh.line.color.rgb = border
        sh.line.width = Pt(border_pt)
    else:
        sh.line.fill.background()
    return sh


def line_h(slide, left, top, width, color=BORDER_SUB, pt=0.75):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(pt))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def oval(slide, left, top, width, height, fill=None, border=BORDER_SUB, border_pt=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, width, height)
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    else:
        sh.fill.background()
    if border:
        sh.line.color.rgb = border
        sh.line.width = Pt(border_pt)
    else:
        sh.line.fill.background()
    return sh


def tb(slide, left, top, width, height, text, size=12, bold=False, color=SLATE_100,
       align=PP_ALIGN.LEFT, italic=False, wrap=True):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf  = box.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.bold   = bold
    r.font.size   = Pt(size)
    r.font.name   = "Segoe UI"
    r.font.color.rgb = color
    r.font.italic = italic
    return box, tf


def mtb(slide, left, top, width, height, lines, wrap=True):
    """Multi-line textbox. lines = list of dicts."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf  = box.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    first = True
    for ln in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = ln.get("align", PP_ALIGN.LEFT)
        if "space_before" in ln:
            p.space_before = Pt(ln["space_before"])
        r = p.add_run()
        r.text = ln.get("text", "")
        r.font.bold   = ln.get("bold", False)
        r.font.size   = Pt(ln.get("size", 12))
        r.font.name   = "Segoe UI"
        r.font.color.rgb = ln.get("color", SLATE_100)
        r.font.italic = ln.get("italic", False)
    return box, tf


def pill(slide, left, top, width, height, text, bg=PANEL_LIGHT, fg=CYAN_PRI, size=10, bold=True):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = bg
    sh.line.fill.background()
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top  = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.bold  = bold
    r.font.size  = Pt(size)
    r.font.name  = "Segoe UI"
    r.font.color.rgb = fg
    return sh


def arrow_right(slide, left, top, width, height, color=SLATE_400):
    sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


# ═══════════════════════════════════════════════════
#  SLIDE 1 — TITLE SLIDE
#  Layout: Dark canvas · orbital rings top-right ·
#          centred wordmark · founder pills row
# ═══════════════════════════════════════════════════
def slide_01_title(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    # Orbital rings (decorative, top-right)
    ring_specs = [
        (Inches(7.5),  Inches(-1.8), Inches(10.0), RGBColor(0x16, 0x28, 0x42), 0.75),
        (Inches(8.2),  Inches(-1.0), Inches(8.0),  RGBColor(0x16, 0x38, 0x58), 0.75),
        (Inches(9.0),  Inches(-0.2), Inches(6.0),  CYAN_PRI,                   1.5),
    ]
    for lx, ly, sz, col, lw in ring_specs:
        oval(s, lx, ly, sz, sz, border=col, border_pt=lw)

    # Dot-grid texture (bottom-right)
    for row in range(7):
        for col in range(6):
            d = s.shapes.add_shape(MSO_SHAPE.OVAL,
                Inches(7.5) + Inches(col * 0.55),
                Inches(3.5) + Inches(row * 0.55),
                Inches(0.06), Inches(0.06))
            d.fill.solid()
            d.fill.fore_color.rgb = RGBColor(0x1E, 0x36, 0x5C)
            d.line.fill.background()

    # Section tag
    pill(s, Inches(1.0), Inches(1.1), Inches(4.4), Inches(0.36),
         "DEEPTECH  ·  SPACE AGTECH  ·  GEO-ECONOMETRICS",
         bg=PANEL_MID, fg=CYAN_PRI, size=9.5)

    # Main wordmark
    tb(s, Inches(1.0), Inches(1.65), Inches(9.0), Inches(1.6),
       "AgriCascade", size=72, bold=True, color=WHITE)

    # Green accent rule
    line_h(s, Inches(1.0), Inches(3.45), Inches(6.5), GREEN_PRI, pt=2.5)

    # Subtitle
    mtb(s, Inches(1.0), Inches(3.65), Inches(9.0), Inches(1.3), [
        {"text": "Space-Based Geo-Econometric DSS for Agricultural",
         "size": 19, "color": SLATE_300},
        {"text": "Risk Intelligence & Grain Market Forecasting",
         "size": 19, "color": SLATE_300, "space_before": 2},
    ])

    # Founding team label
    tb(s, Inches(1.0), Inches(5.05), Inches(5.0), Inches(0.3),
       "FOUNDING TEAM", size=9, bold=True, color=SLATE_400)

    # Founder pills
    founders = ["Karimbay Ali", "Tolebay Ramazan", "Roma Ali", "Iskakova Aizhan"]
    px = Inches(1.0)
    for f in founders:
        pill(s, px, Inches(5.42), Inches(2.15), Inches(0.42),
             f, bg=PANEL_LIGHT, fg=WHITE, size=11, bold=True)
        px += Inches(2.33)

    # Year
    tb(s, Inches(10.8), Inches(6.9), Inches(2.0), Inches(0.4),
       "2026", size=11, bold=True, color=SLATE_600, align=PP_ALIGN.RIGHT)

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 2 — THE PROBLEM
#  Layout: Asymmetric 42/58 split-screen
#          LEFT: Large headline + dominant stat
#          RIGHT: 3 stacked horizontal data rows
# ═══════════════════════════════════════════════════
def slide_02_problem(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    # Left panel
    rect(s, 0, 0, Inches(5.5), H, fill=BG_MID)
    rect(s, 0, 0, Inches(0.07), H, fill=GREEN_PRI)  # green edge

    # Left content
    pill(s, Inches(0.35), Inches(0.55), Inches(1.8), Inches(0.33),
         "THE PROBLEM", bg=PANEL_LIGHT, fg=AMBER, size=9)

    mtb(s, Inches(0.35), Inches(1.1), Inches(4.8), Inches(2.0), [
        {"text": "The Satellite", "size": 36, "bold": True, "color": WHITE},
        {"text": "& Soil Blindspot", "size": 36, "bold": True, "color": WHITE, "space_before": 0},
    ])

    tb(s, Inches(0.35), Inches(3.1), Inches(4.8), Inches(1.5),
       "Current ag-monitoring relies on optical satellites that\n"
       "fail under cloud cover and detect crop stress only\n"
       "after visible leaf yellowing — weeks too late.",
       size=13, color=SLATE_300, wrap=True)

    # Dominant stat box (left panel, bottom)
    rect(s, Inches(0.35), Inches(4.85), Inches(4.7), Inches(1.55),
         fill=RGBColor(0x0B, 0x22, 0x1A), border=GREEN_PRI, border_pt=1.5, corner=True)
    mtb(s, Inches(0.6), Inches(5.0), Inches(4.2), Inches(1.2), [
        {"text": "3–4 Weeks", "size": 34, "bold": True, "color": GREEN_PRI},
        {"text": "of early-warning time systematically lost per season",
         "size": 12, "color": SLATE_300, "space_before": 4},
    ])

    # Right: 3 stacked rows
    issues = [
        {
            "num":   "01",
            "head":  "Optical Blindspot",
            "tag":   "Sentinel-2  ·  Cloud Opacity  ·  Post-Stress Detection",
            "body":  "Optical sensors are fully blind under cloud cover. "
                     "Leaf yellowing (NDVI drop) registers only after irreversible "
                     "physiological damage has occurred — far too late for intervention.",
            "color": AMBER,
        },
        {
            "num":   "02",
            "head":  "Root-Zone Moisture Gap",
            "tag":   "Surface Spectral Indices  ·  0–5 cm Scan Depth Only",
            "body":  "NDVI/NDWI indices capture only the leaf canopy surface. "
                     "Hidden root-level water deficits at 0–100 cm depth are completely invisible, "
                     "removing 3–4 weeks of proactive decision window.",
            "color": CYAN_PRI,
        },
        {
            "num":   "03",
            "head":  "Systemic Cascade Effect",
            "tag":   "Yield Drop  ·  Logistics Freeze  ·  Price Shock",
            "body":  "Missed early signals compound into national-scale disruptions: "
                     "yield collapse, elevator throughput bottlenecks, and violent grain "
                     "price spikes on commodity exchanges.",
            "color": RED_ALERT,
        },
    ]

    row_h   = Inches(2.0)
    row_gap = Inches(0.18)
    ry0     = Inches(0.55)

    for i, iss in enumerate(issues):
        ry = ry0 + i * (row_h + row_gap)
        rect(s, Inches(5.75), ry, Inches(7.35), row_h,
             fill=PANEL_DARK, border=BORDER_SUB, border_pt=0.75, corner=True)
        rect(s, Inches(5.75), ry, Inches(0.06), row_h, fill=iss["color"])
        tb(s, Inches(5.95), ry + Inches(0.2), Inches(0.5), Inches(0.45),
           iss["num"], size=11, bold=True, color=iss["color"])
        tb(s, Inches(6.35), ry + Inches(0.15), Inches(6.5), Inches(0.5),
           iss["head"], size=17, bold=True, color=WHITE)
        tb(s, Inches(6.35), ry + Inches(0.65), Inches(6.5), Inches(0.32),
           iss["tag"], size=9.5, bold=True, color=iss["color"])
        tb(s, Inches(6.35), ry + Inches(1.03), Inches(6.5), Inches(0.9),
           iss["body"], size=12, color=SLATE_300, wrap=True)

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 3 — THE SOLUTION
#  Layout: Vertical spine with 3 horizontal feature
#          tiers + right-side stat blocks
# ═══════════════════════════════════════════════════
def slide_03_solution(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    # Header bar
    rect(s, 0, 0, W, Inches(1.55), fill=PANEL_DARK)
    line_h(s, 0, Inches(1.55), W, GREEN_PRI, pt=2.5)

    pill(s, Inches(0.7), Inches(0.38), Inches(1.8), Inches(0.33),
         "THE SOLUTION", bg=RGBColor(0x05, 0x20, 0x15), fg=GREEN_PRI, size=9)
    tb(s, Inches(0.7), Inches(0.82), Inches(10.0), Inches(0.55),
       "AgriCascade Innovation: Pre-Visual Drought Intelligence",
       size=23, bold=True, color=WHITE)

    # Summary tag (top-right)
    rect(s, Inches(10.1), Inches(0.22), Inches(2.95), Inches(1.12),
         fill=RGBColor(0x05, 0x20, 0x15), border=GREEN_PRI, border_pt=1.0, corner=True)
    mtb(s, Inches(10.22), Inches(0.32), Inches(2.7), Inches(0.95), [
        {"text": "Deep-penetration microwave", "size": 9.5, "color": GREEN_PALE, "bold": True},
        {"text": "radiometry + SAR backscatter", "size": 9.5, "color": GREEN_PALE, "space_before": 1},
        {"text": "+ Econometric AI Engine",      "size": 9.5, "color": CYAN_PRI, "bold": True, "space_before": 1},
    ])

    # Vertical spine
    rect(s, Inches(1.5), Inches(1.85), Inches(0.05), Inches(5.4), fill=GREEN_DEEP)

    features = [
        {
            "step":  "01",
            "label": "PRE-VISUAL DROUGHT DETECTION",
            "head":  "Root-Zone Water Deficit Identified Weeks Before NDVI Drop",
            "body":  "Deep microwave radiometry penetrates the soil column to 0–100 cm, measuring "
                     "actual root-zone water reserves. The system raises alerts before any surface-level "
                     "or optical indicator shows distress.",
            "stat":  "+27 Days",
            "stat_l":"Lead time vs. optical detection",
            "color": GREEN_PRI,
            "icon_bg": RGBColor(0x05, 0x20, 0x15),
        },
        {
            "step":  "02",
            "label": "ALL-WEATHER RADAR SENSING",
            "head":  "Sentinel-1 SAR Backscatter — 365 Days/Year, 24/7 Operation",
            "body":  "C-band synthetic aperture radar penetrates clouds, smoke, and darkness. "
                     "Backscatter coefficient (σ°) tracks soil surface dielectric changes and "
                     "crop stem water content continuously, regardless of weather conditions.",
            "stat":  "24/7",
            "stat_l":"All-weather orbital coverage",
            "color": CYAN_PRI,
            "icon_bg": RGBColor(0x02, 0x18, 0x28),
        },
        {
            "step":  "03",
            "label": "ECONOMETRIC MARKET INTELLIGENCE",
            "head":  "Granger-Causal Link: Orbital Anomaly → Grain Price Shock Forecast",
            "body":  "VAR models and Granger causality tests quantify the predictive relationship "
                     "between orbital vegetation anomalies and grain exchange price movements, "
                     "giving commodity desks a probabilistic price outlook 3–4 months out.",
            "stat":  "3–4 Mo",
            "stat_l":"Price forecast horizon",
            "color": AMBER,
            "icon_bg": RGBColor(0x24, 0x18, 0x02),
        },
    ]

    for i, f in enumerate(features):
        fy = Inches(1.85) + i * Inches(1.77)

        # Dot on spine
        oval(s, Inches(1.38), fy + Inches(0.62), Inches(0.24), Inches(0.24),
             fill=f["color"], border=None)

        # Step pill
        pill(s, Inches(1.75), fy + Inches(0.54), Inches(0.6), Inches(0.34),
             f["step"], bg=f["icon_bg"], fg=f["color"], size=10, bold=True)

        # Label
        tb(s, Inches(2.5), fy + Inches(0.56), Inches(7.5), Inches(0.34),
           f["label"], size=9.5, bold=True, color=f["color"])

        # Headline
        tb(s, Inches(1.75), fy + Inches(0.98), Inches(8.25), Inches(0.48),
           f["head"], size=16, bold=True, color=WHITE)

        # Body
        tb(s, Inches(1.75), fy + Inches(1.5), Inches(8.0), Inches(0.65),
           f["body"], size=12, color=SLATE_300, wrap=True)

        # Stat block (right)
        rect(s, Inches(10.3), fy + Inches(0.55), Inches(2.72), Inches(0.95),
             fill=f["icon_bg"], border=f["color"], border_pt=1.0, corner=True)
        mtb(s, Inches(10.4), fy + Inches(0.6), Inches(2.52), Inches(0.85), [
            {"text": f["stat"],   "size": 26, "bold": True, "color": f["color"],  "align": PP_ALIGN.CENTER},
            {"text": f["stat_l"], "size": 9,  "color": SLATE_400, "align": PP_ALIGN.CENTER, "space_before": 3},
        ])

        # Divider
        if i < 2:
            line_h(s, Inches(1.5), fy + Inches(1.67), Inches(11.6), BORDER_SUB, pt=0.6)

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 4 — TECH STACK & DATA PIPELINE
#  Layout: 3 horizontal layer bands (pipeline flow)
#          Each band: label | components | output
# ═══════════════════════════════════════════════════
def slide_04_techstack(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    pill(s, Inches(0.7), Inches(0.38), Inches(2.1), Inches(0.33),
         "TECH ARCHITECTURE", bg=PANEL_LIGHT, fg=CYAN_PRI, size=9)
    tb(s, Inches(0.7), Inches(0.82), Inches(10.0), Inches(0.52),
       "AgriCascade Data Pipeline — Orbital ETL  →  Analytics  →  Interface",
       size=22, bold=True, color=WHITE)
    line_h(s, Inches(0.7), Inches(1.5), Inches(11.9), BORDER_SUB, pt=0.75)

    layers = [
        {
            "idx":   "LAYER  01",
            "name":  "Orbital Data ETL",
            "color": CYAN_PRI,
            "bg":    RGBColor(0x02, 0x18, 0x28),
            "comps": [
                ("Google Earth Engine", "Cloud-scale satellite data processing"),
                ("Sentinel-1 SAR",      "C-band radar backscatter · 20 m res."),
                ("NASA SMAP L3",        "Microwave radiometry · root-zone SM"),
                ("Sentinel-2 MSI",      "Optical NDVI/NDWI · 10 m resolution"),
            ],
            "out": "Calibrated multi-source raster stacks",
        },
        {
            "idx":   "LAYER  02",
            "name":  "Geo-Econometric Engine",
            "color": AMBER,
            "bg":    RGBColor(0x24, 0x18, 0x02),
            "comps": [
                ("Python 3.11+ / Pandas", "ETL, time-series wrangling"),
                ("Statsmodels VAR",       "Vector Autoregression forecasting"),
                ("Granger Causality",     "Satellite anomaly → price linkage"),
                ("Isolation Forest",      "Anomaly detection & outlier scoring"),
            ],
            "out": "Price shock probabilities & risk scores",
        },
        {
            "idx":   "LAYER  03",
            "name":  "GIS Interface & Logistics",
            "color": GREEN_PRI,
            "bg":    RGBColor(0x05, 0x20, 0x15),
            "comps": [
                ("OSMnx Road Graphs",  "207 elevator logistics routes"),
                ("Streamlit Dashboard","Operational web interface"),
                ("Folium / Leaflet",   "Interactive spatial risk maps"),
                ("AlertBot (Telegram)","Automated early-warning dispatch"),
            ],
            "out": "Live DSS dashboard + automated alerts",
        },
    ]

    band_h  = Inches(1.62)
    band_y0 = Inches(1.65)
    band_gap = Inches(0.18)
    comp_w   = Inches(2.27)
    comp_h   = Inches(0.92)

    for li, layer in enumerate(layers):
        ly = band_y0 + li * (band_h + band_gap)

        rect(s, Inches(0.5), ly, Inches(12.33), band_h,
             fill=layer["bg"], border=layer["border"] if "border" in layer else layer["color"],
             border_pt=0.75)

        # Layer label column
        tb(s, Inches(0.65), ly + Inches(0.12), Inches(1.5), Inches(0.38),
           layer["idx"], size=8.5, bold=True, color=layer["color"])
        tb(s, Inches(0.65), ly + Inches(0.48), Inches(1.5), Inches(0.95),
           layer["name"], size=13, bold=True, color=WHITE, wrap=True)

        # Vertical separator
        rect(s, Inches(2.25), ly + Inches(0.18), Inches(0.04), Inches(1.25), fill=layer["color"])

        # Component boxes
        for ci, (cname, cdetail) in enumerate(layer["comps"]):
            cx = Inches(2.45) + ci * (comp_w + Inches(0.2))
            rect(s, cx, ly + Inches(0.18), comp_w, comp_h,
                 fill=BG_DARK, border=BORDER_SUB, border_pt=0.5, corner=True)
            mtb(s, cx + Inches(0.12), ly + Inches(0.24), comp_w - Inches(0.24), comp_h, [
                {"text": cname,   "size": 11,  "bold": True,  "color": WHITE},
                {"text": cdetail, "size": 9.5, "bold": False, "color": SLATE_400, "space_before": 3},
            ])

        # Output
        tb(s, Inches(11.7), ly + Inches(0.18), Inches(1.08), Inches(0.35),
           "OUTPUT →", size=8.5, bold=True, color=layer["color"])
        tb(s, Inches(11.7), ly + Inches(0.56), Inches(1.08), Inches(0.95),
           layer["out"], size=8.5, color=SLATE_300, wrap=True)

    # Bottom note
    tb(s, Inches(0.7), Inches(7.05), Inches(12.0), Inches(0.32),
       "All orbital data ingested via Google Earth Engine API · Python 3.11 · "
       "Target deployment: cloud-native SaaS",
       size=9, color=SLATE_600, italic=True)

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 5 — PRODUCT MODULES
#  Layout: 2×2 asymmetric grid — each cell has
#          distinct accent colour + micro-metric
# ═══════════════════════════════════════════════════
def slide_05_modules(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    rect(s, 0, 0, W, Inches(1.3), fill=PANEL_DARK)
    line_h(s, 0, Inches(1.3), W, BORDER_SUB, pt=1.0)

    pill(s, Inches(0.7), Inches(0.32), Inches(1.8), Inches(0.33),
         "PLATFORM MODULES", bg=PANEL_LIGHT, fg=GREEN_PRI, size=9)
    tb(s, Inches(0.7), Inches(0.73), Inches(9.5), Inches(0.48),
       "Core Product: 4 Integrated Intelligence Modules",
       size=21, bold=True, color=WHITE)

    modules = [
        {
            "id":     "A",
            "name":   "PhenoMap",
            "sub":    "Optical Vegetation Monitor",
            "color":  GREEN_PRI,
            "bg":     RGBColor(0x05, 0x20, 0x15),
            "data":   "Sentinel-2  ·  NDVI / NDWI  ·  10 m resolution",
            "desc":   "Tracks crop biomass and canopy moisture dynamics across the growing "
                      "season. Generates vegetation anomaly z-score maps at 10 m resolution.",
            "metric": "10 m",
            "m_lbl":  "Spatial resolution",
        },
        {
            "id":     "B",
            "name":   "HydroRisk",
            "sub":    "Radar Root-Zone Moisture Mapping",
            "color":  CYAN_PRI,
            "bg":     RGBColor(0x02, 0x18, 0x28),
            "data":   "Sentinel-1 SAR  +  NASA SMAP  ·  C + L band fusion",
            "desc":   "Root-zone soil moisture maps and hydrological stress zone alerts. "
                      "Issues pre-visual warnings 3–4 weeks before optical yellowing.",
            "metric": "0–100 cm",
            "m_lbl":  "Root-zone depth sensing",
        },
        {
            "id":     "C",
            "name":   "SiloSentry",
            "sub":    "Elevator Network & Logistics",
            "color":  AMBER,
            "bg":     RGBColor(0x24, 0x18, 0x02),
            "data":   "OSMnx  ·  207 Elevators  ·  Road & Rail Graphs",
            "desc":   "Spatial network of 207 grain elevators with storage capacity, "
                      "route optimisation, and logistics bottleneck detection.",
            "metric": "207",
            "m_lbl":  "Elevator nodes mapped",
        },
        {
            "id":     "D",
            "name":   "MarketSpread & RiskEngine",
            "sub":    "Econometric AI Price Forecasting",
            "color":  RED_ALERT,
            "bg":     RGBColor(0x22, 0x08, 0x12),
            "data":   "VAR  ·  Granger Causality  ·  Price Shock Probability",
            "desc":   "Probabilistic grain price shock forecasting engine. Quantifies the "
                      "lag structure between orbital anomalies and commodity exchange movements.",
            "metric": "3–4 Mo",
            "m_lbl":  "Price forecast horizon",
        },
    ]

    gw   = Inches(6.1)
    gh   = Inches(2.85)
    gx   = [Inches(0.4), Inches(6.75)]
    gy   = [Inches(1.5), Inches(4.52)]

    for i, m in enumerate(modules):
        col = i % 2
        row = i // 2
        mx, my = gx[col], gy[row]

        rect(s, mx, my, gw, gh, fill=m["bg"], border=m["color"], border_pt=1.0)
        rect(s, mx, my, Inches(0.07), gh, fill=m["color"])  # accent bar

        # Large dimmed ID letter (background)
        bg_r = min(255, (m["bg"][0] + m["color"][0]) // 2)
        bg_g = min(255, (m["bg"][1] + m["color"][1]) // 2)
        bg_b = min(255, (m["bg"][2] + m["color"][2]) // 2)
        tb(s, mx + gw - Inches(1.45), my + Inches(0.02), Inches(1.4), Inches(1.6),
           m["id"], size=70, bold=True, color=RGBColor(bg_r, bg_g, bg_b))

        # Module name + sub
        tb(s, mx + Inches(0.25), my + Inches(0.2), gw - Inches(1.65), Inches(0.55),
           m["name"], size=22, bold=True, color=WHITE)
        tb(s, mx + Inches(0.25), my + Inches(0.76), gw - Inches(1.65), Inches(0.36),
           m["sub"], size=11, bold=True, color=m["color"])

        # Data source
        pill(s, mx + Inches(0.25), my + Inches(1.18), gw - Inches(1.8), Inches(0.3),
             m["data"], bg=BG_DARK, fg=SLATE_400, size=9, bold=False)

        # Description
        tb(s, mx + Inches(0.25), my + Inches(1.58), gw - Inches(1.75), Inches(0.95),
           m["desc"], size=11.5, color=SLATE_300, wrap=True)

        # Metric
        mtb(s, mx + Inches(0.25), my + gh - Inches(0.82), Inches(2.5), Inches(0.75), [
            {"text": m["metric"], "size": 20, "bold": True,  "color": m["color"]},
            {"text": m["m_lbl"],  "size": 9,  "bold": False, "color": SLATE_400, "space_before": 2},
        ])

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 6 — SCIENTIFIC VALIDATION & BACKTESTING
#  Layout: Dominant "27" number (massive, center-stage)
#          + 4-panel fact strip at bottom
# ═══════════════════════════════════════════════════
def slide_06_validation(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    # Ambient glow (large dark-green oval centre)
    oval(s, Inches(3.0), Inches(0.5), Inches(7.5), Inches(6.0),
         fill=RGBColor(0x02, 0x12, 0x09), border=None)

    pill(s, Inches(0.7), Inches(0.38), Inches(2.2), Inches(0.33),
         "SCIENTIFIC VALIDATION", bg=PANEL_LIGHT, fg=GREEN_PRI, size=9)

    # Massive "27" stat
    tb(s, Inches(1.5), Inches(0.65), Inches(10.5), Inches(3.3),
       "27", size=220, bold=True, color=GREEN_PRI, align=PP_ALIGN.CENTER)

    # Label lines
    tb(s, Inches(1.5), Inches(3.55), Inches(10.5), Inches(0.7),
       "DAYS  LEAD  TIME", size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    tb(s, Inches(1.5), Inches(4.27), Inches(10.5), Inches(0.42),
       "Pre-visual drought detection  ·  North Kazakhstan Oblast  ·  2021 Retrospective Experiment",
       size=13, color=SLATE_400, align=PP_ALIGN.CENTER, italic=True)

    # 4-panel fact strip
    facts = [
        {"label": "Test Region",  "value": "North Kazakhstan\n(SKO Oblast)",            "color": CYAN_PRI},
        {"label": "Event Year",   "value": "2021 Severe\nDrought Benchmark",             "color": AMBER},
        {"label": "Data Leakage", "value": "0%\nZero leakage · clean baseline",          "color": GREEN_PRI},
        {"label": "Result",       "value": "Signal flagged 27 days\nbefore official declaration", "color": RED_ALERT},
    ]

    fw  = Inches(2.9)
    fh  = Inches(1.95)
    fx0 = Inches(0.7)
    fy  = Inches(5.35)

    for i, f in enumerate(facts):
        fx = fx0 + i * (fw + Inches(0.3))
        rect(s, fx, fy, fw, fh, fill=PANEL_DARK, border=f["color"], border_pt=1.0)
        rect(s, fx, fy, fw, Inches(0.06), fill=f["color"])
        mtb(s, fx + Inches(0.18), fy + Inches(0.18), fw - Inches(0.36), fh - Inches(0.22), [
            {"text": f["label"], "size": 9.5, "bold": True, "color": f["color"]},
            {"text": f["value"], "size": 14,  "bold": True, "color": WHITE, "space_before": 6},
        ])

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 7 — MARKET OPPORTUNITY (TAM/SAM/SOM)
#  Layout: Concentric rings diagram (left 45%)
#          + detailed breakdown table (right 55%)
# ═══════════════════════════════════════════════════
def slide_07_market(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    pill(s, Inches(0.7), Inches(0.38), Inches(2.0), Inches(0.33),
         "MARKET OPPORTUNITY", bg=PANEL_LIGHT, fg=AMBER, size=9)
    tb(s, Inches(0.7), Inches(0.82), Inches(9.5), Inches(0.52),
       "TAM / SAM / SOM — Space Ag-Analytics Market Sizing",
       size=22, bold=True, color=WHITE)
    line_h(s, Inches(0.7), Inches(1.48), Inches(12.0), BORDER_SUB, pt=0.7)

    # LEFT — Concentric ring diagram
    cx = Inches(2.85)
    cy = Inches(4.05)

    rings = [
        (Inches(4.6), RGBColor(0x0E, 0x35, 0x55), CYAN_PRI, 2.0),
        (Inches(3.1), RGBColor(0x18, 0x28, 0x10), AMBER,    2.5),
        (Inches(1.75),RGBColor(0x05, 0x28, 0x18), GREEN_PRI,3.0),
    ]
    ring_lbls = [
        ("TAM", "$12.0B", CYAN_PRI,  Inches(2.3)),
        ("SAM", "$140M",  AMBER,     Inches(1.55)),
        ("SOM", "$1.2M",  GREEN_PRI, Inches(0.87)),
    ]

    for (sz, bg_col, bord_col, bw), (tag, val, fc, off) in zip(rings, ring_lbls):
        oval(s, cx - sz/2, cy - sz/2, sz, sz, fill=bg_col, border=bord_col, border_pt=bw)
        tb(s, cx + off + Inches(0.06), cy - Inches(0.35), Inches(1.35), Inches(0.72),
           f"{tag}\n{val}", size=11, bold=True, color=fc)

    tb(s, cx - Inches(0.7), cy - Inches(0.38), Inches(1.4), Inches(0.76),
       "3-Year\nTarget", size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # RIGHT — breakdown table
    markets = [
        {
            "tag":   "TAM",
            "val":   "$12.0 Billion",
            "name":  "Total Addressable Market",
            "sub":   "Global Ag-Geoanalytics & Space Monitoring Software Market (CAGR 14%)",
            "color": CYAN_PRI,
        },
        {
            "tag":   "SAM",
            "val":   "$140 Million",
            "name":  "Serviceable Addressable Market",
            "sub":   "Space Ag-Monitoring & DSS — Central Asia & CIS countries",
            "color": AMBER,
        },
        {
            "tag":   "SOM",
            "val":   "$1.2 Million",
            "name":  "Serviceable Obtainable — 3-Year Target",
            "sub":   "207 grain elevators + major agri-holdings in Kazakhstan (Year 1–3)",
            "color": GREEN_PRI,
        },
    ]

    tx0 = Inches(6.5)
    ty  = Inches(1.75)
    th  = Inches(1.67)
    tgap = Inches(0.2)

    for i, m in enumerate(markets):
        my = ty + i * (th + tgap)
        rect(s, tx0, my, Inches(6.5), th, fill=PANEL_DARK, border=m["color"], border_pt=0.8)
        rect(s, tx0, my, Inches(0.06), th, fill=m["color"])

        pill(s, tx0 + Inches(0.18), my + Inches(0.22), Inches(0.85), Inches(0.32),
             m["tag"], bg=BG_DARK, fg=m["color"], size=10, bold=True)
        tb(s, tx0 + Inches(1.15), my + Inches(0.15), Inches(5.15), Inches(0.52),
           m["val"], size=24, bold=True, color=m["color"])
        mtb(s, tx0 + Inches(1.15), my + Inches(0.72), Inches(5.1), Inches(0.9), [
            {"text": m["name"], "size": 13, "bold": True,  "color": WHITE},
            {"text": m["sub"],  "size": 10, "bold": False, "color": SLATE_400, "space_before": 4},
        ])

    tb(s, tx0, Inches(6.95), Inches(6.5), Inches(0.35),
       "Source: MarketsandMarkets · Grand View Research · OECD AgTech Report 2025",
       size=9, color=SLATE_600, italic=True)

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 8 — STRATEGIC ROADMAP
#  Layout: Horizontal milestone timeline (top half)
#          + 3-pillar impact strip (bottom)
# ═══════════════════════════════════════════════════
def slide_08_roadmap(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    pill(s, Inches(0.7), Inches(0.38), Inches(2.0), Inches(0.33),
         "STRATEGIC ROADMAP", bg=PANEL_LIGHT, fg=GREEN_PRI, size=9)
    tb(s, Inches(0.7), Inches(0.82), Inches(10.0), Inches(0.52),
       "Impact & Scaling Trajectory — 2026–2030",
       size=22, bold=True, color=WHITE)
    line_h(s, Inches(0.7), Inches(1.47), Inches(12.0), BORDER_SUB, pt=0.7)

    # Timeline track
    track_y = Inches(2.7)
    line_h(s, Inches(0.6), track_y, Inches(12.0), BORDER_SUB, pt=2.0)

    milestones = [
        {
            "phase": "PHASE 01",
            "year":  "2025–2026",
            "title": "Scientific MVP & Validation",
            "items": ["27-day lead time proven on SKO 2021 event",
                      "PhenoMap + HydroRisk modules live",
                      "Backtesting: 0% data leakage"],
            "color": GREEN_PRI,
        },
        {
            "phase": "PHASE 02",
            "year":  "2026–2027",
            "title": "Pilot: 207 Elevator Network",
            "items": ["SiloSentry logistics integration",
                      "MarketSpread price forecasting v1",
                      "MoAg & AgroHolding pilot contracts"],
            "color": CYAN_PRI,
        },
        {
            "phase": "PHASE 03",
            "year":  "2028",
            "title": "National Scale — Kazakhstan",
            "items": ["KazEOSat constellation integration",
                      "National food security DSS contract",
                      "Grain exchange API live feed"],
            "color": AMBER,
        },
        {
            "phase": "PHASE 04",
            "year":  "2029–2030",
            "title": "CIS & Central Asian Expansion",
            "items": ["Multi-country satellite consortium",
                      "Central Asian grain corridor analytics",
                      "Series A — $5M target"],
            "color": RED_ALERT,
        },
    ]

    mw   = Inches(2.9)
    mx0  = Inches(0.6)

    for i, m in enumerate(milestones):
        mx = mx0 + i * (mw + Inches(0.27))
        dot_x = mx + mw / 2 - Inches(0.18)

        # Node on timeline
        oval(s, dot_x, track_y - Inches(0.19), Inches(0.36), Inches(0.36),
             fill=m["color"], border=None)

        # Alternate cards above/below
        above = (i % 2 == 0)
        card_h = Inches(2.25)
        card_y = track_y - Inches(2.55) if above else track_y + Inches(0.45)
        stub_h = Inches(0.6)

        # Connector stub
        stub_y = track_y - stub_h if above else track_y + Inches(0.19)
        rect(s, dot_x + Inches(0.15), stub_y, Inches(0.04), stub_h, fill=m["color"])

        # Card
        rect(s, mx, card_y, mw, card_h, fill=PANEL_DARK, border=m["color"], border_pt=0.8)
        rect(s, mx, card_y, mw, Inches(0.05), fill=m["color"])

        mtb(s, mx + Inches(0.18), card_y + Inches(0.15), mw - Inches(0.36), Inches(0.85), [
            {"text": m["phase"], "size": 9,  "bold": True, "color": m["color"]},
            {"text": m["year"],  "size": 11, "bold": True, "color": SLATE_300, "space_before": 3},
            {"text": m["title"], "size": 14, "bold": True, "color": WHITE, "space_before": 6},
        ])

        for j, it in enumerate(m["items"]):
            tb(s, mx + Inches(0.18), card_y + Inches(1.05) + j * Inches(0.37),
               mw - Inches(0.36), Inches(0.35),
               "● " + it, size=9.5, color=SLATE_400)

    # Bottom pillar strip
    pillars = [
        {
            "title": "National Food Security",
            "desc":  "Space remote sensing delivers 27-day pre-visual drought warnings to MoAg "
                     "and national agri-holdings, guaranteeing food security resilience.",
            "color": GREEN_PRI,
        },
        {
            "title": "KazEOSat Constellation Integration",
            "desc":  "Data fusion with Kazakhstan's sovereign satellite fleet for independent, "
                     "nationally-controlled orbital intelligence.",
            "color": CYAN_PRI,
        },
        {
            "title": "Grain Exchange Intelligence API",
            "desc":  "Direct API feed to commodity exchanges and trading desks enabling "
                     "data-driven hedging and export quota decisions.",
            "color": AMBER,
        },
    ]

    pw  = Inches(3.9)
    ph  = Inches(1.12)
    px0 = Inches(0.65)
    py  = Inches(6.2)

    for i, p in enumerate(pillars):
        px = px0 + i * (pw + Inches(0.33))
        rect(s, px, py, pw, ph, fill=PANEL_MID, border=BORDER_SUB, border_pt=0.5)
        rect(s, px, py, pw, Inches(0.045), fill=p["color"])
        mtb(s, px + Inches(0.2), py + Inches(0.12), pw - Inches(0.4), ph, [
            {"text": p["title"], "size": 12, "bold": True,  "color": p["color"]},
            {"text": p["desc"],  "size": 9.5,"bold": False, "color": SLATE_400, "space_before": 4},
        ])

    return s


# ═══════════════════════════════════════════════════
#  SLIDE 9 — TEAM & CLOSING
#  Layout: Orbital ring backdrop · centred tagline ·
#          horizontal team strip · bottom contact bar
# ═══════════════════════════════════════════════════
def slide_09_closing(prs):
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    # Orbital rings (centre-aligned)
    for sz in [Inches(10.5), Inches(7.8), Inches(5.2)]:
        oval(s, (W - sz) / 2, (H - sz) / 2, sz, sz,
             border=RGBColor(0x15, 0x28, 0x42), border_pt=0.75)

    # Top accent bar
    rect(s, 0, 0, W, Inches(0.07), fill=GREEN_PRI)

    # Closing tagline
    mtb(s, Inches(1.0), Inches(1.35), Inches(11.3), Inches(1.8), [
        {"text": "Predictive Intelligence for",
         "size": 36, "bold": True, "color": SLATE_300, "align": PP_ALIGN.CENTER},
        {"text": "Agriculture & Grain Logistics",
         "size": 36, "bold": True, "color": WHITE, "align": PP_ALIGN.CENTER, "space_before": 0},
    ])

    # Brand name
    tb(s, Inches(1.0), Inches(3.05), Inches(11.3), Inches(0.9),
       "AgriCascade", size=52, bold=True, color=GREEN_PRI, align=PP_ALIGN.CENTER)

    line_h(s, Inches(3.5), Inches(4.08), Inches(6.3), GREEN_DEEP, pt=1.5)

    # Team strip
    team = [
        {"name": "Karimbay Ali",    "role": "Lead Researcher &\nPlatform Architect"},
        {"name": "Tolebay Ramazan", "role": "Geo-Econometrics &\nData Science"},
        {"name": "Roma Ali",        "role": "GIS Engineering &\nVisualization"},
        {"name": "Iskakova Aizhan", "role": "Scientific Advisor &\nProject Lead"},
    ]

    tw   = Inches(2.9)
    tx0  = Inches(0.88)
    ty   = Inches(4.4)

    for i, tm in enumerate(team):
        tx = tx0 + i * (tw + Inches(0.27))
        col = GREEN_PRI if i < 3 else CYAN_PRI
        rect(s, tx, ty, tw, Inches(0.045), fill=col)
        mtb(s, tx, ty + Inches(0.1), tw, Inches(1.4), [
            {"text": tm["name"], "size": 14, "bold": True,  "color": WHITE},
            {"text": tm["role"], "size": 10, "bold": False, "color": SLATE_400, "space_before": 5},
        ])

    # Bottom bar
    rect(s, 0, Inches(7.05), W, Inches(0.45), fill=PANEL_DARK)
    tb(s, Inches(0.7), Inches(7.1), Inches(8.0), Inches(0.32),
       "AgriCascade  ·  Space DeepTech AgTech  ·  Kazakhstan  ·  2026",
       size=10, color=SLATE_600)
    tb(s, Inches(9.3), Inches(7.1), Inches(3.7), Inches(0.32),
       "github.com/Ali-bit-bot-ux/EcoEconomicAnalyzer",
       size=10, color=CYAN_PRI, align=PP_ALIGN.RIGHT)

    return s


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════
def build_deck():
    prs = new_prs()

    print("[1/9] Title slide...")
    slide_01_title(prs)
    print("[2/9] Problem slide...")
    slide_02_problem(prs)
    print("[3/9] Solution slide...")
    slide_03_solution(prs)
    print("[4/9] Tech Stack slide...")
    slide_04_techstack(prs)
    print("[5/9] Product Modules slide...")
    slide_05_modules(prs)
    print("[6/9] Scientific Validation slide...")
    slide_06_validation(prs)
    print("[7/9] Market Opportunity slide...")
    slide_07_market(prs)
    print("[8/9] Roadmap slide...")
    slide_08_roadmap(prs)
    print("[9/9] Team & Closing slide...")
    slide_09_closing(prs)

    output = r"D:\legion\Roma Ali pitch-deck.pptx"
    prs.save(output)
    print(f"\n[DONE] Saved: {output}")
    print(f"       Total slides: {len(prs.slides)}")
    return output


if __name__ == "__main__":
    build_deck()
