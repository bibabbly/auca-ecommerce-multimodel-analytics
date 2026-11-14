"""
visualize.py - Part 4: static charts for the technical report (output/charts/*.png).

Reads the result tables written by the MongoDB, Spark and integration jobs:
  01_conversion_funnel.png        integration/funnel_overall.csv          (HBase + MongoDB)
  02_category_revenue_profit.png  mongo/revenue_by_category_month.json    (MongoDB pipeline 1)
  03_customer_segments.png        mongo/user_segments.json                (MongoDB pipeline 2)
  04_clv_drivers.png              integration/clv_users.csv               (CLV + engagement)
  05_views_vs_sales.png           products viewed (sessions) vs units sold (transactions)
  06_cohort_heatmap.png           spark/cohort_registration.csv           (Spark cohort job)

Design rules: one axis per chart, thin bars, recessive hairline grid, text in neutral ink
(never in the series colour), a colour-blind-validated two-colour palette (blue / orange),
a single-hue blue ramp for magnitude, and selective direct labels.

Usage:  python src/visualize.py
"""

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from common import DEFAULT_DATA_DIR, PROJECT_ROOT, iter_records  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "output")
CHARTS = os.path.join(OUT, "charts")

# ---- palette (validated: CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # starts at step 250 (>= 2:1)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 26, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 9,
    "text.parse_math": False,  # literal "$" in labels (money), not TeX math
})


def money(x, _=None):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


def source(fig, text):
    fig.text(0.01, -0.04, f"Source: {text}", fontsize=7.5, color=INK_3, ha="left", va="top")


def save(fig, name):
    os.makedirs(CHARTS, exist_ok=True)
    path = os.path.join(CHARTS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.relpath(path, PROJECT_ROOT)}")


# --------------------------------------------------------------------------- 1. funnel
def chart_funnel():
    df = pd.read_csv(os.path.join(OUT, "integration", "funnel_overall.csv"))
    labels = ["Session started", "Viewed a product", "Added to cart", "Reached checkout", "Purchased"]
    v = df["sessions"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(v))[::-1]
    ax.barh(y, v, height=0.55, color=ORDINAL_BLUE, zorder=2)
    for yi, val, i in zip(y, v, range(len(v))):
        pct = 100 * val / v[0]
        step = "" if i == 0 else f"   ({100 * val / v[i - 1]:.0f}% of previous step)"
        ax.text(val + v[0] * 0.01, yi, f"{val:,}  ·  {pct:.1f}%{step}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, v[0] * 1.55)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}K"))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Conversion funnel: half of shoppers drop out at each of the last two steps")
    subtitle(ax, "300,000 sessions over 90 days. Funnel stages from HBase clickstream, purchases from MongoDB orders")
    ax.set_xticks(np.arange(0, v[0] + 1, 100_000))
    ax.spines["bottom"].set_bounds(0, v[0])
    source(fig, "integration/funnel_clv.py (HBase user_sessions JOIN MongoDB transactions on session_id)")
    save(fig, "01_conversion_funnel.png")


# --------------------------------------------------------------------------- 2. category revenue vs profit
def chart_category():
    rows = json.load(open(os.path.join(OUT, "mongo", "revenue_by_category_month.json")))
    df = pd.DataFrame(rows)
    df = df[df["month"] <= "2026-10"]  # drop the partial spill-over month
    agg = df.groupby("category_name")[["net_revenue", "est_profit"]].sum()
    agg["margin"] = agg["est_profit"] / agg["net_revenue"]
    top = agg.sort_values("net_revenue", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    y = np.arange(len(top))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, top["net_revenue"], height=h, color=BLUE, label="Net revenue", zorder=2)
    ax.barh(y - h / 2 - 0.02, top["est_profit"], height=h, color=ORANGE, label="Estimated gross profit", zorder=2)
    for yi, (_, r) in zip(y, top.iterrows()):
        ax.text(r["est_profit"] + 4e4, yi - h / 2 - 0.02, f"{r['margin']:.0%} margin", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(y, top.index)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, top["net_revenue"].max() * 1.32)
    ax.legend(loc="lower right", handlelength=1, handleheight=0.8)
    best = agg["est_profit"].idxmax()
    ax.set_title("Revenue is flat across categories, profit is not")
    subtitle(ax, f"Top 12 categories by net revenue, Aug-Oct 2026. Most profitable overall: {best} "
                 f"({agg.loc[best, 'margin']:.0%} margin)")
    source(fig, "MongoDB aggregation pipeline 1 ($unwind items, $lookup product margin, $group by category)")
    save(fig, "02_category_revenue_profit.png")


# --------------------------------------------------------------------------- 3. segments
def chart_segments():
    seg = json.load(open(os.path.join(OUT, "mongo", "user_segments.json")))["segments"]
    df = pd.DataFrame(seg).rename(columns={"_id": "segment"})
    df["label"] = df["segment"].str.replace(r"^\d\. ", "", regex=True)
    df = df.iloc[::-1]
    df["user_share"] = df["users"] / df["users"].sum()
    df["rev_share"] = df["revenue"] / df["revenue"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(df))
    for ax, col, color, title in ((axes[0], "user_share", BLUE, "Share of customers"),
                                  (axes[1], "rev_share", ORANGE, "Share of revenue")):
        ax.barh(y, df[col], height=0.55, color=color, zorder=2)
        for yi, v in zip(y, df[col]):
            lbl = "0%" if v == 0 else (f"{v:.0%}" if v >= 0.005 else "<1%")
            ax.text(v + 0.01, yi, lbl, va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 0.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=10, pad=6, fontweight="normal", color=INK_2)
    axes[0].set_yticks(y, df["label"])
    loyal = df[df["label"].str.startswith("Loyal")].iloc[0]
    fig.suptitle(f"Loyal customers (7-11 orders) are {loyal.user_share:.0%} of the base "
                 f"and {loyal.rev_share:.0%} of revenue",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=1.04)
    fig.text(0.0, 0.955, "RFM-style segments of 10,000 customers from the embedded users.purchase_summary "
                         "(at risk = no order in the last 45 days)", fontsize=9.5, color=INK_2)
    source(fig, "MongoDB aggregation pipeline 2 ($switch segment + $facet)")
    save(fig, "03_customer_segments.png")


# --------------------------------------------------------------------------- 4. CLV drivers
def chart_clv():
    df = pd.read_csv(os.path.join(OUT, "integration", "clv_users.csv"))
    drivers = {"Number of orders": "orders", "Average order value": "aov", "Profit margin of basket": "margin",
               "Page views": "page_views", "Number of sessions": "sessions", "Avg. session length": "avg_session_s"}
    corr = pd.Series({k: df[v].corr(df["clv_12m"]) for k, v in drivers.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    y = np.arange(len(corr))
    colors = [ORANGE if k in ("Page views", "Number of sessions", "Avg. session length") else BLUE for k in corr.index]
    ax.barh(y, corr.values, height=0.55, color=colors, zorder=2)
    for yi, v in zip(y, corr.values):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, corr.index)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Pearson correlation with predicted 12-month CLV")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Purchase behaviour (MongoDB)"),
                       Patch(color=ORANGE, label="On-site engagement (HBase)")],
              loc="lower right", handlelength=1, handleheight=0.8)
    q = df.groupby("clv_tier")["clv_12m"].sum()
    ax.set_title("Customer value is driven by repeat purchases, not time on site")
    subtitle(ax, f"Top CLV quartile holds {q['Q4'] / q.sum():.0%} of predicted value; "
                 f"avg CLV ${df[df.clv_tier == 'Q4'].clv_12m.mean():,.0f} vs ${df[df.clv_tier == 'Q1'].clv_12m.mean():,.0f} "
                 "in the bottom quartile")
    source(fig, "integration/funnel_clv.py (MongoDB orders + product margins, HBase session engagement, joined in Spark)")
    save(fig, "04_clv_drivers.png")


# --------------------------------------------------------------------------- 5. views vs sales
def product_views():
    cache = os.path.join(OUT, "spark", "product_views.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    print("  counting product views from session files (one-off, cached) ...")
    views = Counter()
    for s in iter_records(DEFAULT_DATA_DIR, "sessions"):
        for pv in s["page_views"]:
            if pv["page_type"] == "product_detail" and pv["product_id"]:
                views[pv["product_id"]] += 1
    df = pd.DataFrame(views.items(), columns=["product_id", "views"])
    df.to_csv(cache, index=False)
    return df


def chart_views_vs_sales():
    views = product_views()
    units = Counter()
    for t in iter_records(DEFAULT_DATA_DIR, "transactions"):
        for it in t["items"]:
            units[it["product_id"]] += it["quantity"]
    df = views.merge(pd.DataFrame(units.items(), columns=["product_id", "units"]), on="product_id", how="outer").fillna(0)
    top_sold = set(df.nlargest(10, "units")["product_id"])
    top_viewed = set(df.nlargest(10, "views")["product_id"])
    r = df["views"].corr(df["units"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    base = df[~df.product_id.isin(top_sold | top_viewed)]
    ax.scatter(base["views"], base["units"], s=9, color=INK_3, alpha=0.35, linewidths=0, zorder=2,
               label="Other products")
    tv = df[df.product_id.isin(top_viewed)]
    ts = df[df.product_id.isin(top_sold)]
    ax.scatter(tv["views"], tv["units"], s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3,
               label="10 most viewed")
    ax.scatter(ts["views"], ts["units"], s=46, color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=4,
               label="10 best-selling")
    overlap = len(top_sold & top_viewed)
    ax.set_xlabel("Product detail page views (sessions, HBase)")
    ax.set_ylabel("Units sold (transactions, MongoDB)")
    ax.legend(loc="lower right", handletextpad=0.3, framealpha=0.9, frameon=True, edgecolor=GRID)
    ax.set_title("Most-viewed and best-selling are different products")
    both = "no product is" if overlap == 0 else f"{overlap} product{'s are' if overlap > 1 else ' is'}"
    subtitle(ax, f"5,000 products. Correlation between views and units sold r = {r:.2f}; "
                 f"{both} in both top-10 lists")
    source(fig, "session page views vs transaction line items (90 days)")
    save(fig, "05_views_vs_sales.png")


# --------------------------------------------------------------------------- 6. cohort heatmap
def chart_cohorts():
    df = pd.read_csv(os.path.join(OUT, "spark", "cohort_registration.csv"))
    df = df[df["order_month"] <= "2026-10"]
    pv = df.pivot(index="registration_month", columns="order_month", values="revenue_per_member")
    pct = df.pivot(index="registration_month", columns="order_month", values="pct_buying")
    size = df.groupby("registration_month")["cohort_users"].first()
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    # fixed $2,000-$3,000 scale: keeps the ~13% spread visible without exaggerating it
    im = ax.imshow(pv.values, cmap=cmap, aspect="auto", vmin=2000, vmax=3000)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isnan(v):
                continue
            frac = (v - im.norm.vmin) / (im.norm.vmax - im.norm.vmin)
            ax.text(j, i, f"${v:,.0f}\n{pct.values[i, j]:.0f}% buying", ha="center", va="center", fontsize=8,
                    color="white" if frac > 0.3 else INK)
    ax.set_xticks(range(pv.shape[1]), [pd.to_datetime(m).strftime("%b %Y") for m in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"{pd.to_datetime(m).strftime('%b %Y')}  (n={size[m]:,})" for m in pv.index])
    ax.set_xlabel("Order month")
    ax.set_ylabel("Registration cohort")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[2000, 2250, 2500, 2750, 3000],
                      format=FuncFormatter(lambda x, _: f"${x:,.0f}"))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.set_label("Revenue per cohort member", color=INK_2, fontsize=8.5)
    ax.set_title("Every registration cohort spends about the same each month")
    subtitle(ax, "Revenue per cohort member and % of the cohort placing an order. No tenure decay in the synthetic data")
    source(fig, "Spark cohort_analysis.py (registration-month cohorts, partial Nov 2026 excluded)")
    save(fig, "06_cohort_heatmap.png")


def main():
    print(f"Writing charts to {CHARTS}")
    chart_funnel()
    chart_category()
    chart_segments()
    chart_clv()
    chart_views_vs_sales()
    chart_cohorts()


if __name__ == "__main__":
    main()
