# Battery Dispatch Optimizer

A flexible, user-friendly tool for simulating and optimizing battery energy storage dispatch using real or synthetic time series data. Includes an interactive Streamlit web app and batch processing capabilities.

---

## Features
- **Interactive Web UI** (Streamlit):
  - Upload your own time series data (CSV/Excel) or use a provided template
  - Configure battery specs, grid connection, and load profiles
  - Run optimizations for revenue, resilience, or trade-off analysis
  - Visualize dispatch, state of charge, grid flows, and revenue
  - Download results as CSV or ZIP
- **Batch Mode:**
  - Run multiple scenarios programmatically and export results
- **Extensible Core:**
  - Modular, well-documented Python codebase for custom workflows

---

## Folder Structure
```
Battery_Dispatch/
  dispatch_core/         # Core logic (config, data IO, optimization, reporting)
    __init__.py
    config.py
    data_io.py
    optimize.py
    profiles.py
    reporting.py
  streamlit_app.py       # Main Streamlit web app
  run_batch.py           # Batch runner script
  requirements.txt       # Python dependencies
  template_may2015.csv   # Example data file
```

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Battery_Dispatch
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Interactive Web App
Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```
- Open the provided local URL in your browser.
- Upload your data or use the template.
- Configure parameters in the sidebar.
- Click "Run optimisation" to see results and download outputs.

### 2. Batch Mode
Edit `run_batch.py` to set your data file and parameter grid, then run:
```bash
python run_batch.py
```
Results will be saved to `batch_results.csv`.

---

## Data Format Requirements
- **Required columns:**
  - `Datetime` (parseable, e.g. 2023-01-01 00:00)
  - `Load (MW)`
  - `Market Price ($/MWh)`
- **Optional columns:**
  - `Wind (MW)`, `Solar (MW)`, `NatGas (MW)` (filled with zeros if missing)
- **File types:** CSV, Excel (.xlsx)
- See `template_may2015.csv` for an example.

---

## Customization & Extensibility
- All core logic is in `dispatch_core/` and is modular and well-documented.
- You can add new optimization modes, reporting features, or data sources as needed.

---

## Contact
For questions, suggestions, or contributions, please contact:
- [Your Name/Team]
- [Your Email or GitHub]

---

Enjoy optimizing your battery dispatch! 

---

## Fixed Schedule Mode

**New Feature:** You can now manually specify a fixed charge/discharge schedule for the battery, instead of running optimization.

- **How to use:**
  - Select 'Fixed Schedule' in the sidebar mode selector.
  - Enter 'C' for charge, 'D' for discharge, or leave blank for idle in the editable grid.
  - The model will charge/discharge at the maximum allowed rate, limited by battery power, energy, and round-trip efficiency (RTE).
  - No optimization is performed in this mode.

- **Results:**
  - You will see a dispatch overview plot, revenue, shift %, cycles, and other key metrics.
  - Download the full time-series results as CSV.
  - A summary note will indicate that this run used a fixed schedule and enforced all battery constraints.

- **Why use this?**
  - Explore the impact of custom/manual battery schedules.
  - Test scenarios and visualize results without optimization.

--- 