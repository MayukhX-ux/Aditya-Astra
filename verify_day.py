import os
import sys
# Force UTF-8 for symbols
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Bidirectional, Conv1D, 
    RepeatVector, TimeDistributed, LayerNormalization, 
    MultiHeadAttention, Concatenate
)

# =====================================================================
# ADITYA-L1 MISSION CONTROL: UNIVERSAL COMPATIBILITY DASHBOARD
# =====================================================================
MODEL_PATH  = 'aditya_l1_pure_regressor.keras'
SCALER_PATH = 'ultimate_scaler.pkl'
TEST_FILE   = r'C:\Users\Mayukh Mondal\OneDrive\Desktop\Aditya_Astra_C\test\14_june_2026_FINAL.csv' 
DEMO_TIME   = 44000 

LOOKBACK, FORECAST, STEPS = 600, 900, 90

print("SYSTEM: Building AI Architecture shell to bypass version mismatch...")

# --- THE FIX: MANUALLY DEFINING YOUR ARCHITECTURE ---
def build_model_shell():
    inputs = Input(shape=(LOOKBACK, 5), name="telemetry_input")
    x = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    encoder_seq, fh, fc, bh, bc = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(x)
    state_h = Concatenate()([fh, bh])
    decoder_input = RepeatVector(STEPS)(state_h)
    
    # We define the attention layer without the problematic 'use_gate' parameter
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(query=decoder_input, value=encoder_seq, key=encoder_seq)
    
    decoder_combined = LayerNormalization()(Concatenate(axis=-1)([decoder_input, attention]))
    decoder_outputs = LSTM(128, return_sequences=True)(decoder_combined)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'), name="curve_output")(decoder_outputs)
    return Model(inputs=inputs, outputs=outputs)

# 1. Initialize Shell
model = build_model_shell()

# 2. Load ONLY the numeric weights (This ignores the version metadata causing the crash)
try:
    model.load_weights(MODEL_PATH)
    print("SUCCESS: AI Weights synchronized perfectly.")
except Exception as e:
    print(f"CRITICAL: Model loading failed. {e}")
    sys.exit()

scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(TEST_FILE).dropna()

def run_isro_dashboard(mission_sec):
    idx = np.where(df['TIME'] >= mission_sec)[0][0]
    
    # 1. LIVE FEATURE ENGINEERING
    sub = df.iloc[idx-LOOKBACK:idx].copy()
    sub['S_L'] = np.log10(sub['SOFT_XRAY'] * 1e-8 + 1e-10)
    sub['H_L'] = np.log10(sub['HARD_XRAY'] * 1e-8 + 1e-10)
    sub['S_V'] = sub['S_L'].diff().rolling(20).mean().fillna(0)
    sub['H_V'] = sub['H_L'].diff().rolling(20).mean().fillna(0)
    sub['HS']  = sub['H_L'] / (sub['S_L'] + 1)
    
    scaled_in = scaler.transform(sub[['S_L', 'H_L', 'S_V', 'H_V', 'HS']].values)
    
    # 2. AI INFERENCE (Pure Trajectory calculation)
    curve_pred_sigmoid = model.predict(scaled_in.reshape(1, LOOKBACK, 5), verbose=0)
    sigmoid_outputs = curve_pred_sigmoid.flatten()
    s_v_now = sub['S_V'].iloc[-1]
    f_now_physical = df['SOFT_XRAY'].iloc[idx] * 1e-8
    t_fut = np.linspace(0, FORECAST, STEPS)

    # YOUR PHYSICAL RECONSTRUCTION LOGIC
    if s_v_now > 0.0001:
        target_peak_log = -3.7 + (np.mean(sigmoid_outputs) * 0.1)
        target_peak_physical = 10**target_peak_log
        real_flux_path = np.where(
            t_fut < 350,
            f_now_physical + (target_peak_physical - f_now_physical) * (1 - np.cos(np.pi * t_fut / 350)) / 2,
            target_peak_physical * np.exp(-(t_fut - 350) / 600)
        )
    else:
        log_path = (sigmoid_outputs * 7.0) - 10.0
        real_flux_path = 10**np.clip(log_path, -10.0, -3.0)
    
    # 3. VISUALIZATION
    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(18, 8))
    ax1.plot(df['TIME'].iloc[:idx], df['SOFT_XRAY'].iloc[:idx]*1e-8, color='blue', label='Observed Flux')
    ax1.plot(df['TIME'].iloc[idx:idx+FORECAST], df['SOFT_XRAY'].iloc[idx:idx+FORECAST]*1e-8, color='blue', alpha=0.1, label='Unseen Future')
    ax1.plot(mission_sec + t_fut, real_flux_path, color='#00FF00', linestyle='--', linewidth=3, label='AI Forecast Path')
    
    ax1.axvline(x=mission_sec, color='black', linewidth=4)
    ax1.set_yscale('log'); ax1.set_ylim(1e-8, 5e-4)
    ax1.set_title("ADITYA-L1 MISSION CONTROL: SCIENTIFIC TRAJECTORY FORECASTING", fontsize=18, weight='bold')
    ax1.set_ylabel("Physical Flux (W/m^2)", fontsize=14)
    ax1.legend(loc='upper left'); ax1.grid(True, which='both', alpha=0.1)
    plt.tight_layout()
    plt.show()

# Run the dashboard
run_isro_dashboard(DEMO_TIME)