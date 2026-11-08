import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = ['EMPTY', 'SITTING', 'LYING']
SETS = {'CSI-only': 'csi_features.csv', 'Radar-only': 'radar_features.csv', 'Fusion': 'fusion_features.csv'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='features')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics, all_predictions = [], []
    for model_name, filename in SETS.items():
        df = pd.read_csv(data_dir / filename).sort_values(['round','stage','window']).reset_index(drop=True)
        feature_cols = [c for c in df if c not in ('round','stage','window')]
        X, y, groups = df[feature_cols], df['stage'], df['round']
        pred = np.empty(len(df), dtype=object)
        cv = GroupKFold(n_splits=5)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
            pipe = make_pipeline(
                SimpleImputer(strategy='median', add_indicator=True),
                RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
            )
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = pipe.predict(X.iloc[te])
        accuracy = accuracy_score(y, pred)
        macro_f1 = f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)
        metrics.append({'model':model_name,'accuracy':accuracy,'macro_f1':macro_f1})
        print(f'{model_name}: accuracy={accuracy:.3f}, macro-F1={macro_f1:.3f}')
        print(classification_report(y, pred, labels=LABELS, zero_division=0))
        report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(out / f'{model_name.lower().replace("-", "_")}_report.csv')
        cm = confusion_matrix(y, pred, labels=LABELS)
        fig, ax = plt.subplots(figsize=(5.5,4.6))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model_name} — grouped CV')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f'{model_name.lower().replace("-", "_")}_confusion_matrix.png', dpi=160)
        plt.close(fig)
        part = df[['round','stage','window']].copy()
        part['model'] = model_name
        part['predicted_stage'] = pred
        all_predictions.append(part)
    pd.DataFrame(metrics).to_csv(out / 'model_comparison.csv', index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / 'cv_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.5,4.2))
    ax.bar([m['model'] for m in metrics], [m['macro_f1'] for m in metrics])
    ax.set_ylim(0,1)
    ax.set_ylabel('Macro-F1 (grouped cross-validation)')
    ax.set_title('Posture classification comparison')
    for i, m in enumerate(metrics):
        ax.text(i, m['macro_f1']+0.02, f"{m['macro_f1']:.3f}", ha='center')
    fig.tight_layout()
    fig.savefig(out / 'model_comparison.png', dpi=160)
    plt.close(fig)
    print(f'Results saved to {out.resolve()}')

if __name__ == '__main__':
    main()
