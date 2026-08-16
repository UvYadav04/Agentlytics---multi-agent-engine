"""One-time local generator for the sample/dummy files shipped alongside this script
(dummy.csv, dummy.xlsx, dummy.pdf) - see shared/dummy_files.py for how they're used.

dummy.csv is already checked into this directory and is the single source of truth for the
underlying data; this script just re-derives dummy.xlsx and dummy.pdf from it so all three stay
consistent. Run it once whenever dummy.csv changes:

    cd Server/analyzerEngine/sample_data
    pip install pandas openpyxl reportlab
    python generate_samples.py

Not run automatically at deploy/startup time - these are static assets meant to be committed
alongside the app, not regenerated on every boot.
"""
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(HERE, "dummy.csv"), parse_dates=["date"])


def write_xlsx(df: pd.DataFrame) -> None:
    path = os.path.join(HERE, "dummy.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sales", index=False)
        summary = (
            df.groupby(["region", "product"], as_index=False)
            .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
        )
        summary.to_excel(writer, sheet_name="Summary by Region+Product", index=False)
    print(f"wrote {path}")


def write_pdf(df: pd.DataFrame) -> None:
    path = os.path.join(HERE, "dummy.pdf")
    total_revenue = int(df["revenue"].sum())
    total_units = int(df["units_sold"].sum())

    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    top_region, top_region_revenue = by_region.index[0], int(by_region.iloc[0])

    by_product = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    top_product, top_product_revenue = by_product.index[0], int(by_product.iloc[0])

    by_month = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()
    first_month_rev, last_month_rev = int(by_month.iloc[0]), int(by_month.iloc[-1])
    growth_pct = round((last_month_rev - first_month_rev) / first_month_rev * 100, 1)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=letter, title="Sample Sales Summary")
    story = [
        Paragraph("Q1-Q2 2026 Sales Summary (Sample Data)", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "This is a sample document bundled with the app so new users have something real "
            "to ask questions about before uploading their own files.",
            styles["Italic"],
        ),
        Spacer(1, 16),
        Paragraph(
            f"Across January-April 2026, total revenue across all regions and products reached "
            f"${total_revenue:,}, on {total_units:,} units sold. Monthly revenue grew from "
            f"${first_month_rev:,} in the first month to ${last_month_rev:,} in the most recent "
            f"month, a {growth_pct}% increase.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Paragraph(
            f"{top_region} was the strongest region, generating ${top_region_revenue:,} in total "
            f"revenue. {top_product} was the best-selling product line by revenue, contributing "
            f"${top_product_revenue:,} over the period.",
            styles["BodyText"],
        ),
        Spacer(1, 20),
        Paragraph("Revenue by Region", styles["Heading2"]),
        Spacer(1, 8),
    ]

    table_data = [["Region", "Total Revenue"]] + [
        [region, f"${int(rev):,}"] for region, rev in by_region.items()
    ]
    table = Table(table_data, colWidths=[2.5 * inch, 2.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#CC785C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5EFE9")]),
    ]))
    story.append(table)

    doc.build(story)
    print(f"wrote {path}")


if __name__ == "__main__":
    data = load_data()
    write_xlsx(data)
    write_pdf(data)
