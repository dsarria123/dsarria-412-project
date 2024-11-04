# DSC412-project


# Wave Prediction for Pipeline using LSTM and 51101 Bouy data

This project aims to predict surf conditions (wave height, period, and direction) using historical and real-time data from NOAA buoys. By leveraging a Long Short-Term Memory (LSTM) model, the goal is to provide surfers with reliable, spot-specific wave forecasts by incorporating wave transit time adjustments to translate offshore buoy data to nearshore conditions.

## Setup


### Creating the Virtual Environment

To set up the environment, follow these steps:

1. **Navigate** to the project directory:
   ```bash
   cd /path/to/project
   ```

2. **Create the virtual environment** named "surf":
   ```bash
   python -m venv surf
   ```

3. **Activate the environment**:
   - On Windows:
     ```bash
     .\surf\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source surf/bin/activate
     ```

4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This command will install necessary libraries  required to run the LSTM model and build the dashboard.

5. **Deactivate** the environment (when done) using:
   ```bash
   deactivate
   ```

---

## Project Overview
This project focuses on creating a wave prediction model to assist surfers by forecasting wave height, period, and direction specifically for surf spots near NOAA buoy stations. The model uses historical meteorological and wave data from **NOAA Station 51101**.

### Methodology

1. **Data Collection**: 
   - Historical data from NOAA is processed, including meteorological and wave parameters (e.g., wind speed, wave height).
   - The dataset spans multiple years to capture seasonal and yearly variations.

2. **Data Preprocessing**:
   - Data is standardized and cleaned, with timestamps created for each entry.
   - Wave transit time adjustments are applied to align the buoy data with the specific surf spot conditions.

3. **Feature Engineering NOTDONE**:
   - Lagged features and rolling averages are calculated to capture trends in wave characteristics over time.

4. **Model Development**:
   - An LSTM neural network is trained on past wave and weather data to predict future surf conditions at the surf spot.

5. **Real-Time Integration NOTDONE**:
   - A real-time forecasting system can be set up to pull current data from NOAA, updating predictions as new observations are received.

### Implementation Details

- **Tools & Libraries**:
  - **Python**: Core programming and model development.
  - **Pandas & NumPy**: For data cleaning and manipulation.
  - **Keras (TensorFlow)**: To build and train the LSTM model.
  - **Streamlit/Dash**: For creating a simple dashboard for displaying predictions.
  - **NOAA API**: To access historical and real-time buoy data.

---

## Running the Model

1. **Activate the virtual environment**:
   ```bash
   source surf/bin/activate
   ```

2. **Run the code blocks in the model training ipynb (model.ipynb)**:

