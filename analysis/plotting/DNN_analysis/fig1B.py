import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

today_date = datetime.now().strftime("%m_%d")

csv_path = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05.csv'
save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/count_elecs'
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

main_temporal_rois = ['STG', 'STS', 'MTG', 'HG', 'PT', 'PP']
dev_categories = ['Early childhood', 'Middle childhood', 'Early adolescence', 'Late adolescence']

dev_colors = {
    "Early childhood":   "#98df8a",
    "Middle childhood":  "#009c41",
    "Early adolescence": "#3393ff",
    "Late adolescence":  "#021ca4"
}

legend_labels = {
    "Early childhood":   "Early Childhood (age 4-5)",
    "Middle childhood":  "Middle Childhood (age 6-11)",
    "Early adolescence": "Early Adolescence (age 12-17)",
    "Late adolescence":  "Late Adolescence (age 18-21)"
}

def configure_border(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Compute grouped counts
temporal_dev_df = df[df['short_anat'].isin(main_temporal_rois)]

grouped_dev_roi = (
    temporal_dev_df.groupby(['short_anat', 'dev'])
    .size()
    .unstack(fill_value=0)
    .reindex(main_temporal_rois)
    .T
    .reindex(dev_categories)
    .T
)

grouped_melt = grouped_dev_roi.reset_index().melt(
    id_vars='short_anat',
    var_name='Developmental Stage',
    value_name='Count'
)

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
sns.set_style("ticks")

sns.barplot(
    data=grouped_melt,
    x='short_anat',
    y='Count',
    hue='Developmental Stage',
    hue_order=dev_categories,
    palette=dev_colors,
    ax=ax
)

# Labels on bars (skip zeros)
for container in ax.containers:
    labels = [f'{int(v.get_height())}' if v.get_height() > 0 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='edge', padding=2, fontsize=12)

ax.set_ylim(0, 200)
ax.set_yticks(range(0, 201, 50))
ax.set_xlabel("Temporal Lobe ROI", fontsize=11)
ax.set_ylabel("Electrode Count", fontsize=11)
ax.tick_params(axis='both', labelsize=9)
configure_border(ax)
ax.grid(False)

# Legend outside
handles, labels = ax.get_legend_handles_labels()
new_labels = [legend_labels[l] for l in labels]
ax.legend(
    handles, new_labels,
    title="Development Stage",
    title_fontsize=10,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0.
)

ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

ax.tick_params(axis='both', colors='black')          # tick labels
ax.tick_params(axis='both', which='both', color='black')  # tick marks

ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

legend = ax.get_legend()
legend.get_title().set_color('black')
for text in legend.get_texts():
    text.set_color('black')
    
plt.tight_layout()

out_png = os.path.join(save_dir, f'grouped_roi_dev_counts_{today_date}.png')
out_pdf = os.path.join(save_dir, f'grouped_roi_dev_counts_{today_date}.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved:\n  {out_png}\n  {out_pdf}")