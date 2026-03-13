import argparse
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

WINDOW_SIZE = 125       
N_CHANNELS  = 9       
STRIDE      = 62       
THRESHOLD   = 0.5      
MODEL_PATH  = "models/fall_detector.keras"
SCALER_PATH = "data/processed/scaler.pkl"


def load_session(filepath):
    df = pd.read_csv(filepath)
    expected = ["timestamp_ms",
                "waist_x", "waist_y", "waist_z",
                "thigh_x", "thigh_y", "thigh_z",
                "event_flag"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"[deploy_test] Missing columns in CSV: {missing}")

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    return df


def run_inference(model, scaler, df, threshold):
    sensor_cols = ["waist_x", "waist_y", "waist_z", "waist_gx", "waist_gy", "waist_gz",
                   "thigh_x", "thigh_y", "thigh_z"]
    data   = df[sensor_cols].values     
    events = df["event_flag"].values    
    n      = len(data)

    results = []
    for start in range(0, n - WINDOW_SIZE, STRIDE):
        end    = start + WINDOW_SIZE
        window = data[start:end]       

        
        window_flat   = window.reshape(-1, N_CHANNELS)
        window_scaled = scaler.transform(window_flat)
        window_scaled = window_scaled.reshape(1, WINDOW_SIZE, N_CHANNELS)

        prob       = float(model.predict(window_scaled, verbose=0)[0][0])
        prediction = int(prob > threshold)

        mid_start = start + int(WINDOW_SIZE * 0.25)
        mid_end   = start + int(WINDOW_SIZE * 0.75)
        actual    = 1 if events[mid_start:mid_end].sum() > 0 else 0

        results.append({
            "start":      start,
            "end":        end,
            "prob":       prob,
            "prediction": prediction,
            "actual":     actual
        })
    return results


def print_timeline(results, sample_rate=50):

    print(f"\n{'Time':>8}  {'Prob':>7}  {'Pred':>6}  {'Actual':>8}  Status")
    print(f"{'─' * 60}")

    tp = fp = tn = fn = 0

    for r in results:
        time_sec = r["start"] / sample_rate
        pred     = r["prediction"]
        actual   = r["actual"]
        prob     = r["prob"]
        if   pred == 1 and actual == 1:
            status = "  CAUGHT FALL"
            tp += 1
        elif pred == 1 and actual == 0:
            status = " FALSE ALARM"
            fp += 1
        elif pred == 0 and actual == 0:
            status = "   no fall"
            tn += 1
        elif pred == 0 and actual == 1:
            status = " MISSED FALL  ←— CRITICAL"
            fn += 1

        if pred == 1 or actual == 1:
            print(f"{time_sec:>7.1f}s  {prob:>7.4f}  {pred:>6}  {actual:>8}  {status}")

    print(f"\n{'─' * 60}")
    print(f"  Caught falls   (TP) : {tp}")
    print(f"  Missed falls   (FN) : {fn}  <- must be zero for deployment")
    print(f"  False alarms   (FP) : {fp}")
    print(f"  Clean windows  (TN) : {tn}")

    total_falls = tp + fn
    if total_falls > 0:
        recall = tp / total_falls
        print(f"\n  Session recall      : {recall:.2%}")

    print()
    if fn > 0:
        print(f"     {fn} missed fall(s) — NOT ready for deployment.")
        print(f"     Options:")
        print(f"       1. Lower --threshold (e.g. try 0.3)")
        print(f"       2. Collect more fall training data")
        print(f"       3. Re-examine class weights in train.py")
    else:
        print(f"     No missed falls. Model ready for STM32 deployment.")
        if fp > 5:
            print(f"     High false alarm rate ({fp} FP). Consider raising --threshold slightly.")


def sweep_thresholds(results):
    print(f"\n[deploy_test] Threshold sweep for this session:")
    print(f"  {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'FP Count':<10} {'FN Count'}")
    print(f"  {'─' * 56}")

    probs   = np.array([r["prob"]   for r in results])
    actuals = np.array([r["actual"] for r in results])

    for thresh in np.arange(0.1, 1.0, 0.1):
        preds = (probs > thresh).astype(int)

        tp = int(((preds == 1) & (actuals == 1)).sum())
        fp = int(((preds == 1) & (actuals == 0)).sum())
        fn = int(((preds == 0) & (actuals == 1)).sum())

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        flag = " <- zero missed falls" if fn == 0 else ""
        print(f"  {thresh:<12.1f} {recall:<10.2%} {precision:<12.2%} {fp:<10} {fn}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validate fall detection model on a recorded session CSV'
    )
    parser.add_argument('--file',
                        required=True,
                        type=str,
                        help='Path to raw session CSV (e.g. data/raw/20240315_forward_fall.csv)')
    parser.add_argument('--threshold',
                        default=THRESHOLD,
                        type=float,
                        help=f'Decision threshold (default: {THRESHOLD})')
    parser.add_argument('--sweep',
                        action='store_true',
                        help='Also print threshold sweep table')
    args = parser.parse_args()

    print(f"[deploy_test] Model      : {MODEL_PATH}")
    print(f"[deploy_test] Scaler     : {SCALER_PATH}")
    print(f"[deploy_test] Session    : {args.file}")
    print(f"[deploy_test] Threshold  : {args.threshold}\n")

    model  = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = load_session(args.file)
    print(f"[deploy_test] Loaded {len(df)} samples ({len(df) / 50:.1f}s at 50Hz)")

    results = run_inference(model, scaler, df, threshold=args.threshold)
    print(f"[deploy_test] Processed {len(results)} windows\n")

    print_timeline(results)

    if args.sweep:
        sweep_thresholds(results)