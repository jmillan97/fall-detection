import glob
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

#constants
SAMPLE_RATE = 50
WINDOW_SEC = 2.5
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SEC)
OVERLAP = 0.5
STRIDE = int(WINDOW_SIZE * (1-OVERLAP))
LABEL_RATIO = 0.5
CHANNELS = 9

def load_session(filepath):
    #load raw csv from logger
    try:
        df = pd.read_csv(filepath)
        expected = ["timestamp_ms", "waist_x", "waist_y", "waist_z", "thigh_x", "thigh_y"
                    ,"thigh_z", "event_flag"]
        if not all(col in df.columns for col in expected):
            print(f"[WARN] Missing columns in {filepath}")
            return None
        
        df = df.dropna()
        for col in expected:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        return df
    except Exception as e:
        print(f"[ERROR] Could not load {filepath}: {e}")
        return None

def normalize(X):
    n_windows, window_size, n_channels = X.shape

    X_flat = X.reshape(-1, n_channels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(n_windows, window_size, n_channels), scaler

def make_windows(df, session_id):
    sensor_cols = ["waist_x", "waist_y", "waist_z", "thigh_x", "thigh_y"
                    ,"thigh_z"]
    data = df[sensor_cols].values
    events = df["event_flag"].values
    windows = []
    n_samples = len(data)
    for start in range(0, n_samples - WINDOW_SIZE, STRIDE):
        end = start + WINDOW_SIZE

        window = data[start:end]
        segment = events[start:end]

        mid_start = start + int(WINDOW_SIZE * 0.25)
        mid_end = start + int(WINDOW_SIZE * 0.75)
        label = 1 if events[mid_start:mid_end].sum() > 0 else 0
        windows.append(window, label, session_id)
    return windows
def build_dataset(data_dir="data/raw"):
    all_windows = []
    session_ids = []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"[PREPROCESS] Found {len(csv_files)} session files")

    for i, filepath in enumerate(csv_files):
        print(f"\tLoading: {os.path.basename(filepath)}")
        df = load_session(filepath)
        if df is None:
            continue
        windows = make_windows(df, session_id=1)
        all_windows.extend(windows)
        session_ids.append(i)
    
    X = np.array([w[0] for w in all_windows])
    Y = np.array([w[1] for w in all_windows])
    s = np.array([w[2] for w in all_windows]) #session ids
    print(f"\n[PREPROCESS] Total Windows: {len(X)}")
    print(f"\t\tFall windows    : {Y.sum()}")
    print(f"\t\tNon-fall        : {(Y==0).sum()}")
    print(f"\t\tClass balance   : {Y.mean():.2%} positive")

    unique_sessions = np.unique(s)
    test_sessions = unique_sessions[::4]
    test_mask = np.isin(s, test_sessions)
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    Y_Train, Y_test = Y[train_mask], Y[test_mask]
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = build_dataset()
    X_train, scaler = normalize(X_train)
    X_test, _ = normalize(X_test)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/Y_train.npy", Y_train)
    np.save("data/processed/X_test.npy", Y_test)

    import joblib
    joblib.dump(scaler, "data/processed/scaler.pkl")

    print("\n[PREPROCESS] Saved to data/processed/")
    