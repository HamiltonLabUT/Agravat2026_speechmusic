import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from datetime import datetime
import os

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
sns.set_style('ticks')

current_date = datetime.now().strftime('%m_%d')

COLOR_TRAINED   = '#9b59b6'   # purple  – musical training
COLOR_UNTRAINED = '#1e8449'   # green – no training
COLOR_UNKNOWN   = '#cccccc'   # light gray – unknown

eps = 1e-7

csv_path            = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05.csv'
music_training_path = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/music_training.xlsx'
save_dir            = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/musical_training/fig5'

data = pd.read_csv(csv_path)

data = data[(data['speech_music_corrs_DNN'] > 0.0) |
            (data['music_only_corrs_DNN']   > 0.0) |
            (data['speech_only_corrs_DNN']  > 0.0) |
            (data['stacked_corrs_DNN']      > 0.0)]

# Load musical training info
music_df = pd.read_excel(music_training_path)
data['subject_id']     = data['subj_id'].astype(str).str.strip()
music_df['subject_id'] = music_df.iloc[:, 0].astype(str).str.strip()

training_col = None
for col in music_df.columns:
    unique_vals = music_df[col].astype(str).str.upper().unique()
    if 'YES' in unique_vals and 'NO' in unique_vals:
        training_col = col
        print(f" Training column: '{col}'")
        break

if training_col is None:
    raise ValueError("Could not find a YES/NO training column in music_training.xlsx")

music_df['has_training'] = music_df[training_col].astype(str).str.upper() == 'YES'
data = data.merge(music_df[['subject_id', 'has_training']], on='subject_id', how='left')

# Filter to higher-order regions
rois = ['STG', 'STS', 'MTG']
d = data[data['short_anat'].isin(rois)].copy()

# Fisher Z-transform correlations
d['mixed_z'] = np.arctanh(np.clip(d['speech_music_corrs_DNN'], -1 + eps, 1 - eps))
d['speech_z'] = np.arctanh(np.clip(d['speech_only_corrs_DNN'], -1 + eps, 1 - eps))
d['music_z']  = np.arctanh(np.clip(d['music_only_corrs_DNN'],  -1 + eps, 1 - eps))
d['diff_z']   = d['speech_z'] - d['music_z']

d = d.dropna(subset=['age', 'diff_z', 'speech_z', 'music_z'])

print(f"\nElectrodes in {rois}: {len(d)}")
print(f"  Trained:   {(d['has_training'] == True).sum()}")
print(f"  Untrained: {(d['has_training'] == False).sum()}")
print(f"  Unknown:   {d['has_training'].isna().sum()}")

training_groups = [
    (True,  COLOR_TRAINED,   'Musical training'),
    (False, COLOR_UNTRAINED, 'No training'),
]

age_ticks = [4, 5, 6, 7, 8, 10, 12, 15, 20]

panels = [
    {
        'y_col':   'speech_z',
        'ylabel':  'Speech representation\nz(r$_{speech}$)',
        'ylim':    (-0.2, 0.6),
        # 'title':   'Speech only',
    },
    {
        'y_col':   'music_z',
        'ylabel':  'Music representation\nz(r$_{music}$)',
        'ylim':    (-0.2, 0.6),
        # 'title':   'Music only',
    },
    {
        'y_col':   'mixed_z',
        'ylabel':  'Mixed representation\nz(r$_{mixed}$)',
        'ylim':    (-0.2, 0.6),
    },
]

# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
# fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
fig.subplots_adjust(wspace=0.35)

for ax, panel in zip(axes, panels):
    ax.set_xscale('log')
    y_col = panel['y_col']

    for has_training, color, label in training_groups:
        mask   = d['has_training'] == has_training
        sub    = d[mask]
        age_v  = sub['age']
        y_v    = sub[y_col]

        ax.scatter(age_v, y_v,
           c=color, alpha=0.45, s=12,
           edgecolors='none', label=label, zorder=3)

        if len(sub) > 2:
            log_age = np.log(age_v)
            z_fit   = np.polyfit(log_age, y_v, 1)
            x_line  = np.linspace(age_v.min(), age_v.max(), 200)
            ax.plot(x_line, np.poly1d(z_fit)(np.log(x_line)),
                c=color, linewidth=1.2, linestyle='--', alpha=0.95, zorder=4)

            r, pval = stats.pearsonr(log_age, y_v)
            n = len(sub)
            # print(f"  [{panel['title'][:1]}] {label}: r={r:.3f}, p={pval:.4f}, n={n}")

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)

    ax.set_xticks(age_ticks)
    ax.set_xticklabels([str(int(a)) for a in age_ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel(panel['ylabel'], fontsize=13)
    ax.set_ylim(panel['ylim'])
    # ax.set_title(panel['title'], fontsize=13, loc='left', pad=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    leg = ax.legend(frameon=False, fontsize=11)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

plt.tight_layout()

os.makedirs(save_dir, exist_ok=True)
png_path = f'{save_dir}/musical_training_{current_date}.png'
pdf_path = f'{save_dir}/musical_training_{current_date}.pdf'
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path,           bbox_inches='tight')

# plt.show()