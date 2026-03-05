import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import mplcursors

def create_separate_stacked_plots(csv_path, output_dir=None):

    df = pd.read_csv(csv_path)

    roi_order = ['STG', 'STS', 'MTG','HG', 'PT', 'PP']

    color_map = {
        'STG': '#264653',
        'STS': '#e76f51',
        'MTG': '#a78bfa',
        'HG': '#2a9d8f',
        'PT': '#e9c46a',
        'PP': '#f4a261'
    }

    label_font = {'fontsize': 10, 'fontfamily': 'Arial'}
    tick_font  = {'fontsize': 8,  'fontfamily': 'Arial'}

    scatter_props = {'alpha': 1, 's': 13, 'linewidths': 0.5}
    gray_props    = {'alpha': 1, 's': 11, 'color': '#d3d3d3'}

    axis_lim  = (-0.35, 0.8)
    axis_ticks = [-0.3, 0, 0.3, 0.7]

    def style_ax(ax):
        ax.set_xlim(*axis_lim)
        ax.set_ylim(*axis_lim)
        ax.set_xticks(axis_ticks)
        ax.set_yticks(axis_ticks)
        ax.tick_params(labelsize=tick_font['fontsize'])
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(True, alpha=0.1, linewidth=0.3, linestyle='-')
        ax.set_axisbelow(True)
        ax.plot([-0.35, 0.8], [-0.35, 0.7], 'k-', alpha=0.3, linewidth=0.8, zorder=1)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4, linewidth=0.5, zorder=0)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.4, linewidth=0.5, zorder=0)

    def add_scatter(ax, roi, valid_data, x_col, y_col):
        mask_positive = (valid_data[x_col] >= 0) & (valid_data[y_col] >= 0)
        mask_negative = ~mask_positive

        if mask_negative.any():
            neg_scatter = ax.scatter(
                valid_data.loc[mask_negative, x_col],
                valid_data.loc[mask_negative, y_col],
                **gray_props, zorder=2
            )
            neg_scatter.roi_data = valid_data[mask_negative].reset_index(drop=True)
            cursor_neg = mplcursors.cursor(neg_scatter, hover=True)
            @cursor_neg.connect("add")
            def on_hover_neg(sel):
                idx = sel.target.index
                pt = sel.artist.roi_data.iloc[idx]
                sel.annotation.set_text(
                    f"{pt['short_anat']}\nSubj: {pt['subj_id']}\n"
                    f"Ch: {pt['channelnames']}\nX: {pt[x_col]:.3f}\nY: {pt[y_col]:.3f}"
                )
                sel.annotation.get_bbox_patch().set_alpha(0.9)

        if mask_positive.any():
            pos_scatter = ax.scatter(
                valid_data.loc[mask_positive, x_col],
                valid_data.loc[mask_positive, y_col],
                color=color_map[roi], **scatter_props, zorder=3
            )
            pos_scatter.roi_data = valid_data[mask_positive].reset_index(drop=True)
            cursor_pos = mplcursors.cursor(pos_scatter, hover=True)
            @cursor_pos.connect("add")
            def on_hover_pos(sel):
                idx = sel.target.index
                pt = sel.artist.roi_data.iloc[idx]
                sel.annotation.set_text(
                    f"{pt['short_anat']}\nSubj: {pt['subj_id']}\n"
                    f"Ch: {pt['channelnames']}\nX: {pt[x_col]:.3f}\nY: {pt[y_col]:.3f}"
                )
                sel.annotation.get_bbox_patch().set_alpha(0.9)

    today_datetime = datetime.today().strftime('%m_%d')

    # =====================================================================
    # FIGURE 1: 4 scatter columns + 1 label column
    # Col 0 (label) | Col 1: Mixed vs Speech | Col 2: Stacked vs Speech |
    #               | Col 3: Stacked vs Mixed | Col 4: Music vs Speech
    # =====================================================================
    comparisons_fig1 = [
        ('speech_music_corrs_DNN', 'speech_only_corrs_DNN', 'Mixed model (r)',   'Speech-separated\nmodel (r)'),
        ('stacked_corrs_DNN',      'speech_only_corrs_DNN', 'Stacked model (r)', 'Speech-separated\nmodel (r)'),
        ('stacked_corrs_DNN',      'speech_music_corrs_DNN','Stacked model (r)', 'Mixed model (r)'),
        ('music_only_corrs_DNN',   'speech_only_corrs_DNN', 'Music-separated\nmodel (r)', 'Speech-separated\nmodel (r)'),
    ]

    n_data_cols = len(comparisons_fig1)
    # width_ratios: narrow label column + 4 data columns
    fig1 = plt.figure(figsize=(13, 11))
    gs1 = fig1.add_gridspec(
        6, n_data_cols + 1,
        width_ratios=[0.18] + [1] * n_data_cols,
        hspace=0.6, wspace=0.1,
        left=0.04, right=0.97, top=0.96, bottom=0.06
    )

    for row_idx, roi in enumerate(roi_order):
        roi_data = df[df['short_anat'] == roi].copy()

        # ROI label column
        ax_label = fig1.add_subplot(gs1[row_idx, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, roi,
                      transform=ax_label.transAxes,
                      fontsize=13, fontweight='bold', fontstyle='italic', color='black',
                      va='center', ha='center', fontfamily='Arial')

        if roi_data.empty:
            continue

        for col_idx, (x_col, y_col, x_label, y_label) in enumerate(comparisons_fig1):
            ax = fig1.add_subplot(gs1[row_idx, col_idx + 1])
            style_ax(ax)
            ax.set_xlabel(x_label, **label_font)
            ax.set_ylabel(y_label, **label_font)

            valid_data = roi_data[roi_data[x_col].notna() & roi_data[y_col].notna()].copy()
            if not valid_data.empty:
                add_scatter(ax, roi, valid_data, x_col, y_col)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for fmt in ['pdf', 'png']:
            fp = os.path.join(output_dir, f'roi_speech_separated_comparisons_{today_datetime}.{fmt}')
            fig1.savefig(fp, dpi=300 if fmt == 'png' else None, format=fmt, bbox_inches='tight')
            print(f"Saved Figure 1: {fp}")

    # =====================================================================
    # FIGURE 2: 2 scatter columns + 1 label column
    # Col 0 (label) | Col 1: Mixed vs Music | Col 2: Stacked vs Music
    # =====================================================================
    comparisons_fig2 = [
        ('speech_music_corrs_DNN', 'music_only_corrs_DNN', 'Mixed model (r)',   'Music-separated\nmodel (r)'),
        ('stacked_corrs_DNN',      'music_only_corrs_DNN', 'Stacked model (r)', 'Music-separated\nmodel (r)'),
    ]

    n_data_cols2 = len(comparisons_fig2)
    fig2 = plt.figure(figsize=(7, 11))
    gs2 = fig2.add_gridspec(
        6, n_data_cols2 + 1,
        width_ratios=[0.18] + [1] * n_data_cols2,
        hspace=0.6, wspace=0.1,
        left=0.06, right=0.97, top=0.96, bottom=0.06
    )

    for row_idx, roi in enumerate(roi_order):
        roi_data = df[df['short_anat'] == roi].copy()

        # ROI label column
        ax_label = fig2.add_subplot(gs2[row_idx, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, roi,
                      transform=ax_label.transAxes,
                      fontsize=13, fontweight='bold', fontstyle='italic', color='black',
                      va='center', ha='center', fontfamily='Arial')

        if roi_data.empty:
            continue

        for col_idx, (x_col, y_col, x_label, y_label) in enumerate(comparisons_fig2):
            ax = fig2.add_subplot(gs2[row_idx, col_idx + 1])
            style_ax(ax)
            ax.set_xlabel(x_label, **label_font)
            ax.set_ylabel(y_label, **label_font)

            valid_data = roi_data[roi_data[x_col].notna() & roi_data[y_col].notna()].copy()
            if not valid_data.empty:
                add_scatter(ax, roi, valid_data, x_col, y_col)

    if output_dir:
        for fmt in ['pdf', 'png']:
            fp = os.path.join(output_dir, f'roi_music_separated_comparisons_{today_datetime}.{fmt}')
            fig2.savefig(fp, dpi=300 if fmt == 'png' else None, format=fmt, bbox_inches='tight')
            print(f"Saved Figure 2: {fp}")

    plt.show()
    return fig1, fig2


if __name__ == "__main__":
    csv_path   = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05.csv'
    output_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/ROI/roi_fig3'
    fig1, fig2 = create_separate_stacked_plots(csv_path, output_dir)