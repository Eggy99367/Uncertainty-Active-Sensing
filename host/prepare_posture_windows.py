import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STAGES = ['EMPTY', 'SITTING', 'LYING']
ROI = (-1000, 1000, 600, 2100)
WINDOW_SECONDS = 2

def load(path):
    df = pd.read_csv(path)
    df = df[df['stage'].isin(STAGES) & df['round'].between(1, 10)].copy()
    stage_offset = {'EMPTY': 15, 'SITTING': 60, 'LYING': 105}
    df['stage_elapsed_s'] = df['elapsed_s'] - (df['round']-1)*135 - df['stage'].map(stage_offset)
    df = df[df['stage_elapsed_s'].between(0, 30, inclusive='left')].copy()
    df['window'] = np.floor(df['stage_elapsed_s'] / WINDOW_SECONDS).astype(int)
    return df

def main(data_dir):
    c = load(data_dir/'csi_features.csv')
    r = load(data_dir/'radar_targets.csv')
    p = load(data_dir/'radar_presence.csv')
    keys = ['round','stage','window']
    csi = c.groupby(keys).agg(
        csi_samples=('mean_amplitude','size'),
        csi_amplitude_mean=('mean_amplitude','mean'),
        csi_amplitude_std=('mean_amplitude','std'),
        csi_amplitude_min=('mean_amplitude','min'),
        csi_amplitude_max=('mean_amplitude','max'),
        csi_rolling_variance_mean=('rolling_variance','mean'),
        csi_rolling_variance_std=('rolling_variance','std'),
        csi_rssi_mean=('rssi_dbm','mean'),
        csi_rssi_std=('rssi_dbm','std'),
        csi_invalid_ratio=('first_word_invalid','mean'),
    ).reset_index()
    x0,x1,y0,y1=ROI
    r['in_roi'] = ((r['detected']==1) & r['x_mm'].between(x0,x1) & r['y_mm'].between(y0,y1))
    valid = r[r['in_roi']].copy()
    valid['abs_speed'] = valid['speed_cm_s'].abs()
    radar = valid.groupby(keys).agg(
        radar_roi_target_count=('target_id','size'),
        radar_distance_mean=('distance_mm','mean'),
        radar_distance_std=('distance_mm','std'),
        radar_distance_min=('distance_mm','min'),
        radar_x_mean=('x_mm','mean'),
        radar_y_mean=('y_mm','mean'),
        radar_abs_speed_mean=('abs_speed','mean'),
        radar_resolution_mean=('resolution_mm','mean'),
    ).reset_index()
    frames = r.groupby(keys)['esp_ms'].nunique().rename('radar_frames').reset_index()
    roi_frames = valid.groupby(keys)['esp_ms'].nunique().rename('radar_roi_frames').reset_index()
    radar = frames.merge(roi_frames, on=keys, how='left').merge(radar,on=keys,how='left')
    radar['radar_roi_frames'] = radar['radar_roi_frames'].fillna(0)
    radar['radar_roi_frame_ratio'] = radar['radar_roi_frames'] / radar['radar_frames'].replace(0,np.nan)
    presence = p.groupby(keys).agg(
        presence_samples=('filtered_human_present','size'),
        presence_fraction=('filtered_human_present','mean'),
        raw_roi_targets_mean=('raw_targets_in_roi','mean'),
    ).reset_index()
    grid = pd.MultiIndex.from_product([range(1,11),STAGES,range(15)],names=keys).to_frame(index=False)
    out=grid.merge(csi,on=keys,how='left').merge(radar,on=keys,how='left').merge(presence,on=keys,how='left')
    out['window_start_s'] = out['window']*WINDOW_SECONDS
    out['window_end_s'] = (out['window']+1)*WINDOW_SECONDS
    for col in ['radar_roi_target_count','radar_roi_frames','radar_roi_frame_ratio','presence_fraction','raw_roi_targets_mean']:
        out[col] = out[col].fillna(0)
    out['radar_has_roi_target'] = (out['radar_roi_frames']>0).astype(int)
    out.to_csv(data_dir/'posture_windows_2s.csv',index=False)
    print('Saved:',data_dir/'posture_windows_2s.csv')
    print('Shape:',out.shape)
    print('Windows by stage:',out.groupby('stage').size().to_dict())
    print('Missing CSI windows:',int(out['csi_samples'].isna().sum()))
    print('Missing radar frames windows:',int(out['radar_frames'].isna().sum()))
    print('Windows without valid ROI radar targets:',int((out['radar_has_roi_target']==0).sum()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path('.'))
    main(ap.parse_args().data_dir)