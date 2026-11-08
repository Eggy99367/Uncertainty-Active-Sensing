import argparse
from pathlib import Path
import json
import pandas as pd

CSI = [
    'csi_amplitude_mean', 'csi_amplitude_std', 'csi_amplitude_min',
    'csi_amplitude_max', 'csi_rolling_variance_mean',
    'csi_rolling_variance_std', 'csi_rssi_mean', 'csi_rssi_std',
]
RADAR = [
    'radar_roi_frame_ratio', 'radar_roi_target_count',
    'radar_distance_mean', 'radar_distance_std', 'radar_distance_min',
    'radar_x_mean', 'radar_y_mean', 'radar_abs_speed_mean',
    'radar_resolution_mean',
]
META = ['round', 'stage', 'window']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='posture_windows_2s.csv')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    needed = META + CSI + RADAR
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df[META].isna().any().any():
        raise ValueError('Missing round/stage/window labels')
    if df.duplicated(META).any():
        raise ValueError('Duplicate round/stage/window keys')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csi_cols = [c for c in CSI if df[c].nunique(dropna=True) > 1]
    radar_cols = [c for c in RADAR if df[c].nunique(dropna=True) > 1]
    feature_sets = {'csi': csi_cols, 'radar': radar_cols, 'fusion': csi_cols + radar_cols}
    for name, cols in feature_sets.items():
        selected = df[META + cols].copy()
        path = out / f'{name}_features.csv'
        selected.to_csv(path, index=False)
        print(f'{name:6} -> {path}  shape={selected.shape}  '
              f'missing feature values={int(selected[cols].isna().sum().sum())}')

    (out / 'feature_sets.json').write_text(
        json.dumps(feature_sets, indent=2), encoding='utf-8')
    report = pd.DataFrame({
        'feature': CSI + RADAR,
        'missing_count': [int(df[c].isna().sum()) for c in CSI + RADAR],
        'unique_values': [int(df[c].nunique(dropna=True)) for c in CSI + RADAR],
    })
    report.to_csv(out / 'feature_quality.csv', index=False)
    print('\nStages:', df['stage'].value_counts().to_dict())
    print('Rounds:', df['round'].nunique())
    print('Constant features:', report.loc[report.unique_values <= 1, 'feature'].tolist())
    print('NOTE: Do not fill NaNs or standardize on the full dataset before cross-validation.')

if __name__ == '__main__':
    main()
