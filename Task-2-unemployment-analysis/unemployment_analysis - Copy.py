from __future__ import annotations

import os
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path
from tkinter import messagebox, scrolledtext
from typing import Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    rename_map = {
        "estimated unemployment rate (%)": "Estimated Unemployment Rate (%)",
        "estimated employed": "Estimated Employed",
        "estimated labour participation rate (%)": "Estimated Labour Participation Rate (%)",
        "estimated labour participation rate": "Estimated Labour Participation Rate (%)",
        "area": "Area",
        "region": "Region",
        "frequency": "Frequency",
        "date": "Date",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in [c.lower() for c in df.columns]}, inplace=True)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)

    required = [
        "Region",
        "Date",
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    for col in ["Estimated Unemployment Rate (%)", "Estimated Employed", "Estimated Labour Participation Rate (%)"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df["Month"] = df["Date"].dt.month_name()
    df["Year"] = df["Date"].dt.year
    df["Period"] = df["Date"].apply(lambda x: "Pre-COVID" if x < pd.Timestamp("2020-03-01") else "COVID-19 Period")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def find_dataset_paths(explicit_paths: Optional[Union[str, Sequence[str]]] = None) -> list[Path]:
    if explicit_paths is not None:
        if isinstance(explicit_paths, (str, os.PathLike)):
            explicit_paths = [str(explicit_paths)]
        else:
            explicit_paths = [str(path) for path in explicit_paths]

        found = []
        for item in explicit_paths:
            path = Path(item).expanduser()
            if path.exists() and path.is_file():
                found.append(path)
        if found:
            return found

    candidates = [
        WORKSPACE_DIR / "Unemployment in India.csv",
        WORKSPACE_DIR / "Unemployment_Rate_upto_11_2020.csv",
        WORKSPACE_DIR / "unemployment.csv",
        WORKSPACE_DIR / "unemployment_data.csv",
    ]
    found = [candidate for candidate in candidates if candidate.exists() and candidate.is_file()]
    if found:
        return found

    discovered = []
    for root in [WORKSPACE_DIR, Path(r"c:\Users\Marya\AppData\Local\Temp")]:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            name = path.name.lower()
            if "unemployment" in name and path not in discovered:
                discovered.append(path)
    return discovered


def combine_datasets(datasets: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not datasets:
        raise ValueError("No datasets provided")

    prepared = [prepare_data(df) for df in datasets]
    if len(prepared) == 1:
        return prepared[0].copy()

    combined = []
    for idx, frame in enumerate(prepared):
        frame = frame.copy()
        frame["Data_Source"] = f"dataset_{idx + 1}"
        combined.append(frame)

    combined_df = pd.concat(combined, ignore_index=True)
    combined_df["Region_Date_Key"] = combined_df["Region"].astype(str) + "|" + combined_df["Date"].dt.strftime("%Y-%m-%d")

    grouped = (
        combined_df.groupby("Region_Date_Key", as_index=False)
        .agg(
            Region=("Region", "first"),
            Date=("Date", "first"),
            Estimated_Unemployment_Rate=("Estimated Unemployment Rate (%)", "mean"),
            Estimated_Employed=("Estimated Employed", "mean"),
            Estimated_Labour_Participation_Rate=("Estimated Labour Participation Rate (%)", "mean"),
            Month=("Month", "first"),
            Year=("Year", "first"),
            Period=("Period", "first"),
            Source_Count=("Data_Source", "nunique"),
        )
        .copy()
    )
    grouped = grouped.rename(
        columns={
            "Estimated_Unemployment_Rate": "Estimated Unemployment Rate (%)",
            "Estimated_Employed": "Estimated Employed",
            "Estimated_Labour_Participation_Rate": "Estimated Labour Participation Rate (%)",
        }
    )
    grouped = grouped.drop(columns=["Region_Date_Key"])
    grouped = grouped.sort_values("Date").reset_index(drop=True)
    grouped["Data_Source"] = "combined"
    return grouped


def load_builtin_dataset() -> pd.DataFrame:
    dataset_paths = find_dataset_paths(None)
    if not dataset_paths:
        raise FileNotFoundError("No built-in unemployment CSV was found.")

    frames = []
    for dataset_path in dataset_paths:
        df = pd.read_csv(dataset_path)
        frames.append(df)

    if len(frames) == 1:
        return prepare_data(frames[0])
    return combine_datasets(frames)


def load_dataset(path: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    if path is None:
        return load_builtin_dataset()

    dataset_paths = find_dataset_paths(path)
    if not dataset_paths:
        raise FileNotFoundError("No unemployment CSV was found. Please provide the file path.")

    frames = []
    for dataset_path in dataset_paths:
        df = pd.read_csv(dataset_path)
        frames.append(df)

    if len(frames) == 1:
        return prepare_data(frames[0])
    return combine_datasets(frames)


def get_overall_summary(df: pd.DataFrame) -> dict:
    overall_avg = df["Estimated Unemployment Rate (%)"].mean()
    peak_month = df.loc[df["Estimated Unemployment Rate (%)"].idxmax(), "Date"]
    peak_rate = df["Estimated Unemployment Rate (%)"].max()
    covid_avg = df.loc[df["Period"] == "COVID-19 Period", "Estimated Unemployment Rate (%)"].mean()
    pre_covid_avg = df.loc[df["Period"] == "Pre-COVID", "Estimated Unemployment Rate (%)"].mean()

    seasonal = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean().sort_index()
    highest_season = seasonal.idxmax()
    lowest_season = seasonal.idxmin()

    return {
        "rows": len(df),
        "overall_avg": overall_avg,
        "peak_month": peak_month,
        "peak_rate": peak_rate,
        "pre_covid_avg": pre_covid_avg,
        "covid_avg": covid_avg,
        "seasonal_high": highest_season,
        "seasonal_low": lowest_season,
    }


def get_year_summary(df: pd.DataFrame, year: int) -> dict:
    if "Year" not in df.columns:
        temp_df = df.copy()
        temp_df["Date"] = pd.to_datetime(temp_df["Date"], dayfirst=True, errors="coerce")
        temp_df = temp_df.dropna(subset=["Date"]).copy()
        temp_df["Year"] = temp_df["Date"].dt.year
        temp_df["Month"] = temp_df["Date"].dt.month_name()
    else:
        temp_df = df.copy()

    year_df = temp_df[temp_df["Year"] == year].copy()
    if year_df.empty:
        raise ValueError(f"No data found for year {year}")

    average_rate = year_df["Estimated Unemployment Rate (%)"].mean()
    monthly_rates = year_df.groupby("Month")["Estimated Unemployment Rate (%)"].mean().round(2)
    monthly_rates = monthly_rates.reindex([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ], fill_value=None)

    return {
        "year": year,
        "average_rate": round(float(average_rate), 2),
        "monthly_rates": monthly_rates.to_dict(),
        "peak_month": year_df.loc[year_df["Estimated Unemployment Rate (%)"].idxmax(), "Month"],
        "peak_rate": round(float(year_df["Estimated Unemployment Rate (%)"].max()), 2),
    }


def plot_trend(df: pd.DataFrame, path: Path) -> None:
    trend = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()
    plt.figure(figsize=(10, 4.5))
    plt.plot(trend["Date"], trend["Estimated Unemployment Rate (%)"], color="#1f77b4", linewidth=2)
    plt.axvline(pd.Timestamp("2020-03-01"), color="red", linestyle="--", label="COVID lockdown")
    plt.title("National Unemployment Trend")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    comparison = df.groupby(["Period", "Region"])["Estimated Unemployment Rate (%)"].mean().reset_index()
    pre = comparison[comparison["Period"] == "Pre-COVID"]
    covid = comparison[comparison["Period"] == "COVID-19 Period"]
    merged = pd.merge(pre, covid, on="Region", suffixes=("_pre", "_covid"))
    if merged.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(merged["Region"], merged["Estimated Unemployment Rate (%)_pre"], color="#b3cde3", label="Pre-COVID")
    plt.bar(merged["Region"], merged["Estimated Unemployment Rate (%)_covid"], color="#fbb4ae", label="COVID-19 Period", alpha=0.85)
    plt.xticks(rotation=45, ha="right")
    plt.title("Regional Unemployment Before vs During COVID-19")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_seasonality(df: pd.DataFrame, path: Path) -> None:
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    seasonal = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean().reindex(month_order)
    plt.figure(figsize=(8, 4.5))
    plt.plot(seasonal.index, seasonal.values, marker="o", color="#2ca02c")
    plt.title("Seasonal Pattern in Unemployment")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_insights(df: pd.DataFrame) -> list[str]:
    summary = get_overall_summary(df)
    insights = []
    if summary["covid_avg"] > summary["pre_covid_avg"]:
        increase = summary["covid_avg"] - summary["pre_covid_avg"]
        insights.append(f"COVID-19 raised the average unemployment rate by {increase:.2f} percentage points compared with pre-COVID levels.")
    else:
        insights.append("The average unemployment rate during COVID-19 was not higher than pre-COVID levels in this dataset.")

    if summary["seasonal_high"] and summary["seasonal_low"]:
        insights.append(f"The highest average unemployment rate appears in {summary['seasonal_high']}, while the lowest is in {summary['seasonal_low']}.")

    worst_regions = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False).head(3)
    regions = ", ".join(worst_regions.index.tolist())
    insights.append(f"The most affected regions in the sample are: {regions}.")
    insights.append("Policy attention should focus on labor-intensive sectors, temporary employment support, and local livelihood programs during future shocks.")
    return insights


def run_analysis(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(exist_ok=True)
    plot_trend(df, output_dir / "unemployment_trend.png")
    plot_comparison(df, output_dir / "pre_vs_covid.png")
    plot_seasonality(df, output_dir / "seasonal_pattern.png")

    summary = get_overall_summary(df)
    insights = generate_insights(df)
    return {"summary": summary, "insights": insights}


def build_report_text(df: pd.DataFrame, result: dict, year: Optional[int] = None, linked_datasets: Optional[list[str]] = None) -> str:
    summary = result["summary"]
    peak_month = summary["peak_month"]
    if hasattr(peak_month, "strftime"):
        peak_date_text = peak_month.strftime("%Y-%m-%d")
    else:
        peak_date_text = str(peak_month)

    lines = []
    lines.append("Unemployment Analysis Report")
    lines.append("=" * 35)
    lines.append(f"Rows analyzed: {summary['rows']}")
    if linked_datasets:
        lines.append(f"Linked datasets used: {', '.join(linked_datasets)}")
    else:
        lines.append("Linked datasets used: 1 file")
    lines.append(f"Average unemployment rate: {summary['overall_avg']:.2f}%")
    lines.append(f"Pre-COVID average: {summary['pre_covid_avg']:.2f}%")
    lines.append(f"COVID-19 average: {summary['covid_avg']:.2f}%")
    lines.append(f"Peak unemployment rate: {summary['peak_rate']:.2f}% on {peak_date_text}")
    lines.append(f"Highest seasonal month: {summary['seasonal_high']}")
    lines.append(f"Lowest seasonal month: {summary['seasonal_low']}")

    if year is not None:
        year_summary = get_year_summary(df, year)
        lines.append("")
        lines.append(f"Year {year} summary:")
        lines.append(f"- Average unemployment rate: {year_summary['average_rate']:.2f}%")
        lines.append(f"- Peak unemployment rate: {year_summary['peak_rate']:.2f}% in {year_summary['peak_month']}")
        lines.append("- Monthly rates:")
        for month, rate in year_summary["monthly_rates"].items():
            if rate is not None:
                lines.append(f"  {month}: {rate:.2f}%")

    lines.append("")
    lines.append("Key insights:")
    for insight in result["insights"]:
        lines.append(f"- {insight}")
    lines.append("")
    lines.append(f"Charts saved in: {OUTPUT_DIR}")
    return "\n".join(lines)


def print_report(df: pd.DataFrame, result: dict) -> None:
    print(build_report_text(df, result))


def run_simple_interface() -> None:
    root = tk.Tk()
    root.title("Unemployment Analysis")
    root.geometry("860x620")
    root.minsize(820, 580)
    root.configure(bg="#f3f6fb")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Card.TFrame", background="#ffffff", relief="flat")
    style.configure("Header.TFrame", background="#102a43")
    style.configure("Action.TButton", background="#2563eb", foreground="#ffffff", padding=(14, 8), font=("Segoe UI", 10, "bold"))
    style.map("Action.TButton", background=[("active", "#1d4ed8"), ("pressed", "#1e40af")], foreground=[("active", "#ffffff")])
    style.configure("Secondary.TButton", background="#e2e8f0", foreground="#0f172a", padding=(10, 8), font=("Segoe UI", 10))
    style.map("Secondary.TButton", background=[("active", "#cbd5e1"), ("pressed", "#94a3b8")], foreground=[("active", "#0f172a")])
    style.configure("TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))
    style.configure("Title.TLabel", background="#102a43", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
    style.configure("Subtitle.TLabel", background="#ffffff", foreground="#475569", font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9))
    style.configure("Input.TEntry", padding=6)

    main_frame = tk.Frame(root, bg="#f3f6fb", padx=16, pady=16)
    main_frame.pack(fill="both", expand=True)

    header = ttk.Frame(main_frame, style="Header.TFrame", padding=(18, 18))
    header.pack(fill="x")
    tk.Label(header, text="Unemployment Analysis Toolkit", bg="#102a43", fg="#ffffff", font=("Segoe UI", 18, "bold"), anchor="w").pack(anchor="w")
    tk.Label(
        header,
        text="Built-in unemployment data is loaded automatically so you can review trends, compare COVID-19 effects, and inspect yearly results.",
        bg="#102a43",
        fg="#e2e8f0",
        font=("Segoe UI", 10),
        wraplength=760,
        justify="left",
    ).pack(anchor="w", pady=(6, 0))

    content = ttk.Frame(main_frame, style="Card.TFrame", padding=(14, 14))
    content.pack(fill="both", expand=True, pady=(12, 0))

    controls = tk.Frame(content, bg="#ffffff")
    controls.pack(fill="x", pady=(0, 10))

    run_button = ttk.Button(controls, text="Run Analysis", style="Action.TButton", command=lambda: run_selected_analysis(root, output_box))
    run_button.pack(side="left")

    year_frame = tk.Frame(content, bg="#ffffff")
    year_frame.pack(fill="x", pady=(0, 10))
    tk.Label(year_frame, text="Inspect a specific year:", bg="#ffffff", fg="#0f172a", font=("Segoe UI", 10, "bold")).pack(side="left")
    year_entry = ttk.Entry(year_frame, width=12)
    year_entry.pack(side="left", padx=(8, 8))
    ttk.Button(year_frame, text="Show Year", style="Secondary.TButton", command=lambda: show_year_summary(root, output_box, year_entry)).pack(side="left")

    output_box = scrolledtext.ScrolledText(
        content,
        height=24,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#f8fafc",
        fg="#0f172a",
        relief="flat",
        padx=10,
        pady=10,
        borderwidth=1,
    )
    output_box.pack(fill="both", expand=True)
    output_box.configure(insertbackground="#0f172a")

    try:
        df = load_builtin_dataset()
        root.dataset = df
        linked_names = [Path(path).name for path in find_dataset_paths(None)]
        root.linked_datasets = linked_names or ["built-in dataset"]
        output_box.insert(tk.END, f"Loaded built-in datasets: {', '.join(linked_names) if linked_names else 'built-in dataset'}\n")
        output_box.insert(tk.END, f"Rows loaded: {len(df)}\n")
        output_box.insert(tk.END, "The built-in datasets are ready. Click 'Run Analysis' to generate the report.\n")
    except Exception as exc:
        root.dataset = None
        root.linked_datasets = None
        output_box.insert(tk.END, f"Unable to load built-in dataset: {exc}\n")

    root.mainloop()


def run_selected_analysis(root: tk.Tk, output_box: scrolledtext.ScrolledText) -> None:
    df = getattr(root, "dataset", None)
    if df is None:
        messagebox.showwarning("No data", "The built-in dataset could not be loaded.")
        return
    try:
        result = run_analysis(df)
        linked_datasets = getattr(root, "linked_datasets", None)
        report = build_report_text(df, result, linked_datasets=linked_datasets)
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, report)
    except Exception as exc:
        messagebox.showerror("Error", f"Analysis failed: {exc}")


def show_year_summary(root: tk.Tk, output_box: scrolledtext.ScrolledText, year_entry: tk.Entry) -> None:
    df = getattr(root, "dataset", None)
    if df is None:
        messagebox.showwarning("No data", "The built-in dataset could not be loaded.")
        return

    year_text = year_entry.get().strip()
    if not year_text.isdigit():
        messagebox.showwarning("Invalid input", "Please enter a valid year.")
        return

    try:
        year = int(year_text)
        result = run_analysis(df)
        linked_datasets = getattr(root, "linked_datasets", None)
        report = build_report_text(df, result, year=year, linked_datasets=linked_datasets)
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, report)
    except Exception as exc:
        messagebox.showerror("Error", f"Unable to show year summary: {exc}")


def main() -> None:
    print("Unemployment Analysis Toolkit")
    print("=" * 32)

    try:
        df = load_dataset(None)
    except Exception as exc:
        print(f"Unable to load built-in dataset: {exc}")
        return

    while True:
        print("\nChoose an option:")
        print("1. Run full analysis")
        print("2. Show summary")
        print("3. Show COVID-19 impact")
        print("4. Show seasonal trend")
        print("5. Show policy insights")
        print("0. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            result = run_analysis(df)
            print_report(df, result)
        elif choice == "2":
            summary = get_overall_summary(df)
            print(f"Overall average unemployment rate: {summary['overall_avg']:.2f}%")
            print(f"Pre-COVID average: {summary['pre_covid_avg']:.2f}%")
            print(f"COVID-19 average: {summary['covid_avg']:.2f}%")
        elif choice == "3":
            summary = get_overall_summary(df)
            print(f"COVID-19 average unemployment rate was {summary['covid_avg']:.2f}% against {summary['pre_covid_avg']:.2f}% before COVID-19.")
        elif choice == "4":
            seasonal = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean().reindex([
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])
            print(seasonal.round(2))
        elif choice == "5":
            for insight in generate_insights(df):
                print(f"- {insight}")
        elif choice == "0":
            print("Thanks for using the toolkit.")
            break
        else:
            print("Please choose a valid option.")


if __name__ == "__main__":
    run_simple_interface()
