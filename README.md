# Neural Networks: Power Consumption Forecast

An end-to-end time-series project that trains an LSTM to forecast the next hour of
electricity demand from the provided `powerconsumption.csv` data.

## Dataset and target

The CSV contains 52,416 observations from Istanbul across three zones, from January
through December 2017. Readings arrive every 10 minutes and include weather features
as well as `PowerConsumption_Zone1`, `PowerConsumption_Zone2`, and
`PowerConsumption_Zone3`. This is not the UCI Individual Household Electric Power
Consumption file: the provided file has three zone-level power series and does not
specify their unit.

The app defaults to the sum of the three zone readings and also lets you select an
individual zone. It averages the source readings into 15-minute bins, uses the
previous 24 hours (96 points) as its input, and predicts four 15-minute steps. The
dashboard labels values in dataset units and estimates daily consumption in
unit-hours instead of assuming the source values are kW.

## Run

Use Python 3.10–3.12. Install the dependencies and launch Streamlit from this folder:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a load series, select a historical forecast anchor, and press **Train LSTM**.
The app trains against the first 80% of the series in chronological order and reports
validation MAE on the remaining portion. The forecast view includes current load,
15-minute and 60-minute predictions, estimated daily consumption, the day's peak,
a recent high-use warning, and an energy note.

## Model notes

The network is a single 48-unit LSTM with dropout and a four-value dense output. It
learns the selected load series only; weather values are not used as model inputs,
since future weather is not available at forecast time. Model scaling is fitted on
the chronological training portion. Forecasts can be anchored at any 15-minute
timestamp with a complete preceding 24-hour window.