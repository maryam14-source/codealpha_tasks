import csv
import math
import os
import sys
import tkinter as tk
from statistics import mean, median
from tkinter import messagebox, ttk

DATA_FILE_CANDIDATES = ["car_price_raw.csv", os.path.expanduser("~/Downloads/car data.csv"), "car_price_data.csv"]
DEFAULT_DATA_FILE = DATA_FILE_CANDIDATES[0]
TARGET = "price"
CURRENT_YEAR = 2026

LEGACY_FEATURES = ["mileage", "age", "horsepower", "engine_size", "fuel_efficiency", "owner_count", "goodwill"]
LEGACY_ENGINEERED_FEATURES = ["mileage_per_age", "power_to_engine", "efficiency_score", "age_squared"]
REAL_BASE_FEATURES = ["age", "driven_kms", "present_price", "owner", "is_dealer", "is_automatic", "fuel_petrol", "fuel_diesel", "fuel_cng", "fuel_other"]
REAL_ENGINEERED_FEATURES = ["km_per_year", "price_per_km", "age_squared"]
MODEL_FEATURES_BY_TYPE = {
    "legacy": LEGACY_FEATURES + LEGACY_ENGINEERED_FEATURES,
    "real": REAL_BASE_FEATURES + REAL_ENGINEERED_FEATURES,
}
INPUT_FIELDS_BY_TYPE = {
    "legacy": [
        {"name": "mileage", "label": "Mileage (km)", "type": "float", "example": "50000"},
        {"name": "age", "label": "Age (years)", "type": "float", "example": "3"},
        {"name": "horsepower", "label": "Horsepower", "type": "float", "example": "120"},
        {"name": "engine_size", "label": "Engine size (L)", "type": "float", "example": "1.6"},
        {"name": "fuel_efficiency", "label": "Fuel efficiency (km/L)", "type": "float", "example": "16"},
        {"name": "owner_count", "label": "Owner count", "type": "float", "example": "2"},
        {"name": "goodwill", "label": "Goodwill score", "type": "float", "example": "7"},
    ],
    "real": [
        {"name": "age", "label": "Age (years)", "type": "float", "example": "6"},
        {"name": "driven_kms", "label": "Driven KMs", "type": "float", "example": "55000"},
        {"name": "present_price", "label": "Present price", "type": "float", "example": "7.87"},
        {"name": "owner", "label": "Owner count", "type": "float", "example": "0"},
        {"name": "fuel_type", "label": "Fuel type", "type": "option", "options": ["Petrol", "Diesel", "CNG", "Other"], "example": "Petrol"},
        {"name": "selling_type", "label": "Selling type", "type": "option", "options": ["Dealer", "Individual"], "example": "Dealer"},
        {"name": "transmission", "label": "Transmission", "type": "option", "options": ["Manual", "Automatic"], "example": "Manual"},
    ],
}


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def resolve_data_path(path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(DATA_FILE_CANDIDATES)

    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.exists(candidate):
                return candidate
            continue

        direct_path = os.path.join(script_dir, candidate)
        if os.path.exists(direct_path):
            return direct_path

        if os.path.exists(candidate):
            return candidate

    return path or DATA_FILE_CANDIDATES[-1]


def normalize_legacy_dataset_row(row):
    return {
        "mileage": parse_float(row.get("mileage", 0.0)),
        "age": parse_float(row.get("age", 0.0)),
        "horsepower": parse_float(row.get("horsepower", 0.0)),
        "engine_size": parse_float(row.get("engine_size", 0.0)),
        "fuel_efficiency": parse_float(row.get("fuel_efficiency", 0.0)),
        "owner_count": parse_float(row.get("owner_count", 0.0)),
        "goodwill": parse_float(row.get("goodwill", 0.0)),
        "price": parse_float(row.get("price", 0.0)),
    }


def normalize_real_dataset_row(row):
    year = parse_int(row.get("Year"), CURRENT_YEAR)
    fuel_type = str(row.get("Fuel_Type", "")).strip().lower()
    selling_type = str(row.get("Selling_type", "")).strip().lower()
    transmission = str(row.get("Transmission", "")).strip().lower()

    return {
        "age": float(max(CURRENT_YEAR - year, 0)),
        "driven_kms": parse_float(row.get("Driven_kms", 0.0)),
        "present_price": parse_float(row.get("Present_Price", 0.0)),
        "owner": parse_float(row.get("Owner", 0.0)),
        "is_dealer": 1.0 if selling_type == "dealer" else 0.0,
        "is_automatic": 1.0 if transmission == "automatic" else 0.0,
        "fuel_petrol": 1.0 if fuel_type == "petrol" else 0.0,
        "fuel_diesel": 1.0 if fuel_type == "diesel" else 0.0,
        "fuel_cng": 1.0 if fuel_type == "cng" else 0.0,
        "fuel_other": 1.0 if fuel_type not in {"petrol", "diesel", "cng"} else 0.0,
        "price": parse_float(row.get("Selling_Price", 0.0)),
    }


def normalize_dataset_row(row):
    headers = {header.strip().lower(): header for header in row.keys()}
    if "selling_price" in headers or "present_price" in headers:
        return normalize_real_dataset_row(row)
    return normalize_legacy_dataset_row(row)


def engineered_features(row):
    if "mileage" in row:
        age = max(float(row.get("age", 0.0)), 1.0)
        engine_size = max(float(row.get("engine_size", 0.0)), 0.1)
        owner_count = max(float(row.get("owner_count", 0.0)), 1.0)
        return {
            "mileage_per_age": float(row.get("mileage", 0.0)) / age,
            "power_to_engine": float(row.get("horsepower", 0.0)) / engine_size,
            "efficiency_score": float(row.get("fuel_efficiency", 0.0)) + float(row.get("goodwill", 0.0)) - owner_count,
            "age_squared": age * age,
        }

    age = max(float(row.get("age", 0.0)), 1.0)
    driven_kms = max(float(row.get("driven_kms", 0.0)), 0.0)
    present_price = max(float(row.get("present_price", 0.0)), 0.0)
    return {
        "km_per_year": driven_kms / age,
        "price_per_km": present_price / max(driven_kms, 1.0),
        "age_squared": age * age,
    }


def build_car_features(car, dataset_type="real"):
    if dataset_type == "legacy":
        return {name: parse_float(car.get(name, 0.0)) for name in LEGACY_FEATURES}

    fuel_type = str(car.get("fuel_type", "")).strip().lower()
    selling_type = str(car.get("selling_type", "")).strip().lower()
    transmission = str(car.get("transmission", "")).strip().lower()

    return {
        "age": parse_float(car.get("age", 0.0)),
        "driven_kms": parse_float(car.get("driven_kms", 0.0)),
        "present_price": parse_float(car.get("present_price", 0.0)),
        "owner": parse_float(car.get("owner", 0.0)),
        "is_dealer": 1.0 if selling_type == "dealer" else 0.0,
        "is_automatic": 1.0 if transmission == "automatic" else 0.0,
        "fuel_petrol": 1.0 if fuel_type == "petrol" else 0.0,
        "fuel_diesel": 1.0 if fuel_type == "diesel" else 0.0,
        "fuel_cng": 1.0 if fuel_type == "cng" else 0.0,
        "fuel_other": 1.0 if fuel_type not in {"petrol", "diesel", "cng"} else 0.0,
    }


def detect_dataset_type(rows):
    if rows and "mileage" in rows[0]:
        return "legacy"
    return "real"


def get_model_feature_names(rows):
    return MODEL_FEATURES_BY_TYPE.get(detect_dataset_type(rows), MODEL_FEATURES_BY_TYPE["legacy"])


def load_dataset(path=None):
    data_path = resolve_data_path(path)
    if os.path.exists(data_path):
        rows = []
        with open(data_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = {header.strip().lower() for header in (reader.fieldnames or [])}
            if "selling_price" in headers or "present_price" in headers:
                rows = [normalize_real_dataset_row(row) for row in reader]
            elif "mileage" in headers and "price" in headers:
                rows = [normalize_legacy_dataset_row(row) for row in reader]
            else:
                rows = [normalize_dataset_row(row) for row in reader]

        if rows:
            return [dict(row, **engineered_features(row)) for row in rows]

    default_rows = [
        {"mileage": 50000, "age": 3, "horsepower": 120, "engine_size": 1.6, "fuel_efficiency": 16, "owner_count": 2, "goodwill": 7, "price": 9500},
        {"mileage": 60000, "age": 4, "horsepower": 110, "engine_size": 1.4, "fuel_efficiency": 15, "owner_count": 3, "goodwill": 6, "price": 8200},
        {"mileage": 30000, "age": 2, "horsepower": 140, "engine_size": 1.8, "fuel_efficiency": 14, "owner_count": 1, "goodwill": 8, "price": 12000},
        {"mileage": 80000, "age": 6, "horsepower": 100, "engine_size": 1.3, "fuel_efficiency": 17, "owner_count": 4, "goodwill": 5, "price": 6500},
        {"mileage": 25000, "age": 1, "horsepower": 160, "engine_size": 2.0, "fuel_efficiency": 13, "owner_count": 1, "goodwill": 9, "price": 15000},
    ]
    return [dict(row, **engineered_features(row)) for row in default_rows]


def preprocess_rows(rows, feature_names, training_stats=None):
    prepared_rows = []
    if training_stats is None:
        medians = {}
        for name in feature_names:
            values = [parse_float(row.get(name, 0.0)) for row in rows]
            medians[name] = median(values) if values else 0.0

        for row in rows:
            prepared = {}
            for name in feature_names:
                value = row.get(name, None)
                prepared[name] = parse_float(value, medians[name]) if value not in (None, "") else medians[name]
            prepared_rows.append(prepared)

        means = {}
        stds = {}
        for name in feature_names:
            values = [row[name] for row in prepared_rows]
            m = mean(values) if values else 0.0
            variance = sum((value - m) ** 2 for value in values) / len(values) if values else 0.0
            s = math.sqrt(variance) if variance > 0 else 1.0
            means[name] = m
            stds[name] = s

        stats = {"medians": medians, "means": means, "stds": stds}
        return prepared_rows, stats

    for row in rows:
        prepared = {}
        for name in feature_names:
            value = row.get(name, None)
            prepared[name] = parse_float(value, training_stats["medians"].get(name, 0.0)) if value not in (None, "") else training_stats["medians"].get(name, 0.0)
        prepared_rows.append(prepared)

    return prepared_rows, training_stats


def standardize_rows(rows, feature_names, training_stats=None):
    if training_stats is None:
        means = {}
        stds = {}
        for name in feature_names:
            values = [row[name] for row in rows]
            m = mean(values) if values else 0.0
            variance = sum((value - m) ** 2 for value in values) / len(values) if values else 0.0
            s = math.sqrt(variance) if variance > 0 else 1.0
            means[name] = m
            stds[name] = s
        standardized = []
        for row in rows:
            new_row = {}
            for name in feature_names:
                new_row[name] = (row[name] - means[name]) / stds[name]
            standardized.append(new_row)
        return standardized, {"means": means, "stds": stds}

    standardized = []
    for row in rows:
        new_row = {}
        for name in feature_names:
            new_row[name] = (row[name] - training_stats["means"][name]) / training_stats["stds"][name]
        standardized.append(new_row)
    return standardized, training_stats


def train_linear_regression(rows, feature_names, target_name, learning_rate=0.008, epochs=18000):
    prepared_rows, stats = preprocess_rows(rows, feature_names)
    standardized_rows, std_stats = standardize_rows(prepared_rows, feature_names, training_stats=None)
    weights = [0.0] * (len(feature_names) + 1)

    for _ in range(epochs):
        for standardized_row, original_row in zip(standardized_rows, rows):
            features = [standardized_row[name] for name in feature_names]
            target = original_row[target_name]
            prediction = weights[0] + sum(w * x for w, x in zip(weights[1:], features))
            error = prediction - target

            weights[0] -= learning_rate * error / len(rows)
            for i in range(len(feature_names)):
                weights[i + 1] -= learning_rate * error * features[i] / len(rows)

    return {
        "weights": weights,
        "feature_names": feature_names,
        "stats": stats,
        "std_stats": std_stats,
    }


def predict_price(model, car, dataset_type="real"):
    weights = model["weights"]
    feature_names = model["feature_names"]
    stats = model["stats"]

    prepared = build_car_features(car, dataset_type=dataset_type)
    prepared.update(engineered_features(prepared))

    standardized = {}
    for name in feature_names:
        standardized[name] = (prepared.get(name, stats["means"].get(name, 0.0)) - stats["means"][name]) / stats["stds"][name]

    total = weights[0]
    for index, name in enumerate(feature_names):
        total += weights[index + 1] * standardized[name]
    return max(0.0, round(total, 2))


def format_price(value):
    return f"{value:,.2f}"


def evaluate_model(rows):
    feature_names = get_model_feature_names(rows)
    split_index = max(2, int(len(rows) * 0.8))
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]
    if len(test_rows) < 1:
        test_rows = rows

    model = train_linear_regression(train_rows, feature_names, TARGET)
    dataset_type = detect_dataset_type(rows)

    actuals = []
    preds = []
    for row in test_rows:
        actual = row[TARGET]
        predicted = predict_price(model, row, dataset_type=dataset_type)
        actuals.append(actual)
        preds.append(predicted)

    if not actuals:
        return {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

    errors = [abs(a - p) for a, p in zip(actuals, preds)]
    squared_errors = [(a - p) ** 2 for a, p in zip(actuals, preds)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mean_actual = sum(actuals) / len(actuals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    r2 = 1 - (sum(squared_errors) / ss_tot) if ss_tot else 1.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 2)}


def format_currency(value):
    return f"₹{value:,.2f} Lakh"


class CarPriceApp:
    def __init__(self, root, data_file=DEFAULT_DATA_FILE):
        self.root = root
        self.root.title("Car Price Predictor")
        self.root.geometry("980x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f4f8")

        self.data_file = resolve_data_path(data_file)
        self.training_data = load_dataset(self.data_file)
        self.dataset_type = detect_dataset_type(self.training_data)
        self.feature_names = get_model_feature_names(self.training_data)
        self.model = train_linear_regression(self.training_data, self.feature_names, TARGET)
        self.metrics = evaluate_model(self.training_data)
        self.input_schema = INPUT_FIELDS_BY_TYPE.get(self.dataset_type, INPUT_FIELDS_BY_TYPE["real"])
        self._build_ui()

    def _build_ui(self):
        container = tk.Frame(self.root, bg="#f0f4f8")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        header = tk.Frame(container, bg="#0f2a44", bd=0, relief="flat")
        header.pack(fill="x", pady=(0, 16))
        tk.Label(header, text="Car Price Predictor", font=("Segoe UI", 22, "bold"), bg="#0f2a44", fg="#ffffff").pack(anchor="w", padx=24, pady=(20, 6))
        tk.Label(header, text="Real-world vehicle value estimation with dataset-aware machine learning.", font=("Segoe UI", 10), bg="#0f2a44", fg="#dce9f9").pack(anchor="w", padx=24, pady=(0, 8))
        tk.Label(header, text="Dynamic input schema • Feature engineering • Regression model • User-friendly predictions", font=("Segoe UI", 9), bg="#0f2a44", fg="#9fc0e3").pack(anchor="w", padx=24, pady=(0, 18))

        body = tk.Frame(container, bg="#f0f4f8")
        body.pack(fill="both", expand=True)

        form_card = tk.Frame(body, bg="#ffffff", bd=1, relief="flat")
        form_card.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=(0, 10))
        tk.Label(form_card, text="Vehicle details", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#16324f").pack(anchor="w", padx=18, pady=(18, 8))
        tk.Label(form_card, text="Enter realistic values for the selected dataset to estimate the market price (prices are in lakhs of rupees).", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7a8f").pack(anchor="w", padx=18, pady=(0, 12))

        self.entries = {}
        for field in self.input_schema:
            row = tk.Frame(form_card, bg="#ffffff")
            row.pack(fill="x", padx=18, pady=6)
            label = tk.Label(row, text=field["label"] + ":", width=20, anchor="w", bg="#ffffff", fg="#2f3f4e")
            label.pack(side="left")
            if field["type"] == "option":
                combo = ttk.Combobox(row, values=field["options"], width=30, state="readonly")
                combo.set(field.get("example", field["options"][0]))
                combo.pack(side="left", fill="x", expand=True)
                self.entries[field["name"]] = combo
            else:
                entry = ttk.Entry(row, width=32)
                entry.insert(0, field.get("example", ""))
                entry.pack(side="left", fill="x", expand=True)
                self.entries[field["name"]] = entry

        button_row = tk.Frame(form_card, bg="#ffffff")
        button_row.pack(fill="x", padx=18, pady=(16, 12))
        predict_button = tk.Button(button_row, text="Predict Price", bg="#1f6feb", fg="#ffffff", font=("Segoe UI", 10, "bold"), relief="flat", padx=18, pady=10, command=self.predict)
        predict_button.pack(side="left")
        clear_button = tk.Button(button_row, text="Clear", bg="#eef4ff", fg="#1e3a5f", font=("Segoe UI", 10), relief="flat", padx=18, pady=10, command=self.clear_fields)
        clear_button.pack(side="left", padx=(10, 0))

        self.result_var = tk.StringVar(value="Prediction will appear here.")
        result_box = tk.Frame(form_card, bg="#eef7ff", bd=1, relief="solid")
        result_box.pack(fill="x", padx=18, pady=(0, 16))
        tk.Label(result_box, textvariable=self.result_var, font=("Segoe UI", 12, "bold"), bg="#eef7ff", fg="#0f4f2d", wraplength=400, justify="center").pack(padx=20, pady=20)

        insights_card = tk.Frame(body, bg="#ffffff", bd=1, relief="flat")
        insights_card.pack(side="right", fill="y", padx=(10, 0), pady=(0, 10))
        tk.Label(insights_card, text="Model insights", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#16324f").pack(anchor="w", padx=18, pady=(18, 8))
        tk.Label(insights_card, text="A polished regression pipeline for dataset-driven pricing.", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7a8f").pack(anchor="w", padx=18, pady=(0, 8))
        tk.Label(insights_card, text="Workflow highlights", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#4b5d6b").pack(anchor="w", padx=18)
        tk.Label(insights_card, text="• Missing values handled with medians\n• Dataset-aware feature engineering\n• Normalized regression inputs\n• Evaluation with MAE, RMSE, and R²", justify="left", bg="#ffffff", fg="#4b5d6b", wraplength=280).pack(anchor="w", padx=18, pady=(8, 14))

        metrics_frame = tk.Frame(insights_card, bg="#f5f9ff", bd=1, relief="solid")
        metrics_frame.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(metrics_frame, text="Evaluation metrics", font=("Segoe UI", 10, "bold"), bg="#f5f9ff", fg="#16324f").pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(metrics_frame, text=f"MAE: {self.metrics['mae']}", bg="#f5f9ff", fg="#0f6b3d").pack(anchor="w", padx=14)
        tk.Label(metrics_frame, text=f"RMSE: {self.metrics['rmse']}", bg="#f5f9ff", fg="#0f6b3d").pack(anchor="w", padx=14)
        tk.Label(metrics_frame, text=f"R²: {self.metrics['r2']}", bg="#f5f9ff", fg="#0f6b3d").pack(anchor="w", padx=14, pady=(0, 12))

        details_frame = tk.Frame(insights_card, bg="#f5f9ff", bd=1, relief="solid")
        details_frame.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(details_frame, text="Dataset summary", font=("Segoe UI", 10, "bold"), bg="#f5f9ff", fg="#16324f").pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(details_frame, text=f"Records: {len(self.training_data)}", bg="#f5f9ff", fg="#315169").pack(anchor="w", padx=14)
        tk.Label(details_frame, text=f"Source: {os.path.basename(self.data_file)}", bg="#f5f9ff", fg="#315169").pack(anchor="w", padx=14, pady=(0, 12))
        tk.Label(details_frame, text="Prices are in lakhs of rupees.", font=("Segoe UI", 9), bg="#f5f9ff", fg="#4b5d6b").pack(anchor="w", padx=14, pady=(0, 12))

        example_box = tk.Frame(insights_card, bg="#eef7ff", bd=1, relief="solid")
        example_box.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(example_box, text="Example input", font=("Segoe UI", 10, "bold"), bg="#eef7ff", fg="#16324f").pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(example_box, text="Age: 6\nDriven KMs: 55000\nPresent price: 7.87 (lakh)\nOwner: 0\nFuel type: Petrol\nSelling type: Dealer\nTransmission: Manual", justify="left", bg="#eef7ff", fg="#315169", wraplength=280).pack(anchor="w", padx=14, pady=(0, 14))

        status_bar = tk.Frame(self.root, bg="#d8e2eb")
        status_bar.pack(fill="x", side="bottom")
        status_label = tk.Label(status_bar, text=f"Loaded {len(self.training_data)} records from {os.path.basename(self.data_file)} — prices shown in lakhs of rupees.", font=("Segoe UI", 9), bg="#d8e2eb", fg="#334155")
        status_label.pack(side="left", padx=16, pady=10)

    def predict(self):
        car = {}
        for field in self.input_schema:
            name = field["name"]
            value = self.entries[name].get()
            if field["type"] == "float":
                try:
                    car[name] = parse_float(value)
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
                    return
            else:
                car[name] = value

        prediction = predict_price(self.model, car, dataset_type=self.dataset_type)
        self.result_var.set(f"Estimated price: {format_currency(prediction)}")

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_var.set("Prediction will appear here.")


def run_cli(data_file=DEFAULT_DATA_FILE):
    training_data = load_dataset(data_file)
    dataset_type = detect_dataset_type(training_data)
    feature_names = get_model_feature_names(training_data)
    model = train_linear_regression(training_data, feature_names, TARGET)
    metrics = evaluate_model(training_data)

    print("Car Price Prediction Model")
    print(f"Loaded {len(training_data)} records from {data_file}")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"R² score: {metrics['r2']}")
    print("Enter details for a new car:")

    if dataset_type == "real":
        car = {
            "age": parse_float(input("Age (years): ")),
            "driven_kms": parse_float(input("Driven KMs: ")),
            "present_price": parse_float(input("Present price: ")),
            "owner": parse_float(input("Owner count: ")),
            "fuel_type": input("Fuel type (Petrol/Diesel/CNG/Other): ").strip(),
            "selling_type": input("Selling type (Dealer/Individual): ").strip(),
            "transmission": input("Transmission (Manual/Automatic): ").strip(),
        }
    else:
        car = {
            "mileage": parse_float(input("Mileage (km): ")),
            "age": parse_float(input("Age (years): ")),
            "horsepower": parse_float(input("Horsepower: ")),
            "engine_size": parse_float(input("Engine size (L): ")),
            "fuel_efficiency": parse_float(input("Fuel efficiency (km/L): ")),
            "owner_count": parse_float(input("Owner count: ")),
            "goodwill": parse_float(input("Goodwill score (1-10): ")),
        }

    prediction = predict_price(model, car, dataset_type=dataset_type)
    print(f"Estimated price: {format_currency(prediction)}")


def main():
    if "--cli" in sys.argv:
        data_file = sys.argv[sys.argv.index("--cli") + 1] if sys.argv.index("--cli") + 1 < len(sys.argv) else DEFAULT_DATA_FILE
        run_cli(data_file)
        return

    data_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FILE
    root = tk.Tk()
    CarPriceApp(root, data_file=data_file)
    root.mainloop()


if __name__ == "__main__":
    main()
