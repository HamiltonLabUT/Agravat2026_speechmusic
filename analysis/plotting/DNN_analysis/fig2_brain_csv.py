import os
os.environ["ETS_TOOLKIT"] = "null"
os.environ["QT_API"] = "pyqt5"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from datetime import datetime
import pyvista as pv
import nibabel as nib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("Script started...")

save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/brain/PiYG_fig2/03_05_26_csv'
os.makedirs(save_dir, exist_ok=True)
print(f"Save directory: {save_dir}")

current_date = datetime.now().strftime('%m_%d')

# Load the precomputed CSV (already filtered, already has selectivity + RGB)
csv_path = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05_r2index_temporallobe.csv'
print(f"Loading electrode data from: {csv_path}")
corr_df_filtered = pd.read_csv(csv_path)
print(f"Total electrode count: {len(corr_df_filtered)}")
print(f"Columns: {list(corr_df_filtered.columns)}")

# Filter to only include target ROIs
target_rois = ['STG', 'STS', 'MTG', 'HG', 'PP', 'PT']
corr_df_filtered = corr_df_filtered[corr_df_filtered['short_anat'].isin(target_rois)].reset_index(drop=True)
print(f"Total electrode count after ROI filter: {len(corr_df_filtered)}")
print(f"Columns: {list(corr_df_filtered.columns)}")

# Set imaging directory
imaging_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_imaging'

# Load brain surfaces
print("Loading brain template surfaces...")
lh_pial_path = os.path.join(imaging_dir, 'cvs_avg35_inMNI152', 'surf', 'lh.pial')
rh_pial_path = os.path.join(imaging_dir, 'cvs_avg35_inMNI152', 'surf', 'rh.pial')

def load_freesurfer_surface(filepath):
    coords, faces = nib.freesurfer.read_geometry(filepath)
    return {'vert': coords, 'tri': faces}

surf_lh = load_freesurfer_surface(lh_pial_path)
surf_rh = load_freesurfer_surface(rh_pial_path)
print("Brain template loaded successfully.")

# Prepare electrode coordinates
elecs = np.array((corr_df_filtered['x'], corr_df_filtered['y'], corr_df_filtered['z'])).T

# COORDINATE SHIFTS
# Shift TCH30 LTG1 (left hemisphere)
tch30_ltg1_mask_prep = (corr_df_filtered['subj_id'] == 'TCH30') & (corr_df_filtered['channel'] == 132)
tch30_ltg1_prep_indices = np.where(tch30_ltg1_mask_prep.values)[0]
if len(tch30_ltg1_prep_indices) > 0:
    x_offset = 50
    original_x = elecs[tch30_ltg1_prep_indices, 0].copy()
    elecs[tch30_ltg1_prep_indices, 0] = original_x - x_offset
    print(f"\nTCH30 LTG1 x: {original_x} -> {original_x - x_offset}")
else:
    print("\nWARNING: TCH30 LTG1 not found!")

# Shift TCH19 LSTG6 (left hemisphere)
tch19_lstg6_mask_prep = (corr_df_filtered['subj_id'] == 'TCH19') & (corr_df_filtered['channelnames'] == 'LSTG6')
tch19_lstg6_prep_indices = np.where(tch19_lstg6_mask_prep.values)[0]
if len(tch19_lstg6_prep_indices) > 0:
    x_offset = 50
    original_x = elecs[tch19_lstg6_prep_indices, 0].copy()
    elecs[tch19_lstg6_prep_indices, 0] = original_x - x_offset
    print(f"\nTCH19 LSTG6 x: {original_x} -> {original_x - x_offset}")
else:
    print("\nWARNING: TCH19 LSTG6 not found!")

# Shift S0022 PST-PI6 (right hemisphere)
s0022_mask_prep = (corr_df_filtered['subj_id'] == 'S0022') & (corr_df_filtered['channelnames'] == 'PST-PI6')
s0022_prep_indices = np.where(s0022_mask_prep.values)[0]
if len(s0022_prep_indices) > 0:
    x_offset = 50
    original_x = elecs[s0022_prep_indices, 0].copy()
    elecs[s0022_prep_indices, 0] = original_x + x_offset
    print(f"\nS0022 PST-PI6 x: {original_x} -> {original_x + x_offset}")
else:
    print("\nWARNING: S0022 PST-PI6 not found!")

# Shift S0005 RParGr3 (right hemisphere)
s0005_mask_prep = (corr_df_filtered['subj_id'] == 'S0005') & (corr_df_filtered['channelnames'] == 'RParGr3')
s0005_prep_indices = np.where(s0005_mask_prep.values)[0]
if len(s0005_prep_indices) > 0:
    x_offset = 50
    original_x = elecs[s0005_prep_indices, 0].copy()
    elecs[s0005_prep_indices, 0] = original_x + x_offset
    print(f"\nS0005 RParGr3 x: {original_x} -> {original_x + x_offset}")
else:
    print("\nWARNING: S0005 RParGr3 not found!")

# PULL PRECOMPUTED VALUES FROM CSV
selectivity = pd.to_numeric(corr_df_filtered['sp_mu_stack_idx'], errors='coerce').fillna(0.0).values
print(selectivity)

stacked     = corr_df_filtered['stacked_corrs_DNN'].values
speech_only = corr_df_filtered['speech_only_corrs_DNN'].values
music_only  = corr_df_filtered['music_only_corrs_DNN'].values

# Magnitude for dot sizing: best of stacked/speech/music per electrode
magnitude = np.maximum(np.maximum(
    np.nan_to_num(stacked, nan=0.0),
    np.nan_to_num(speech_only, nan=0.0)),
    np.nan_to_num(music_only, nan=0.0))

print(f"\nSelectivity index range: {np.nanmin(selectivity):.3f} to {np.nanmax(selectivity):.3f}")
print(f"Mean selectivity: {np.nanmean(selectivity):.3f}")
print(f"Magnitude range:  {np.min(magnitude):.3f} to {np.max(magnitude):.3f}")

speech_selective = np.sum(selectivity > 0.3)
music_selective  = np.sum(selectivity < -0.3)
both_selective   = np.sum(np.abs(selectivity) <= 0.3)
print(f"\nElectrode counts by selectivity:")
print(f"  Speech selective (>0.3):   {speech_selective}")
print(f"  Music selective (<-0.3):   {music_selective}")
print(f"  Both/Mixed (-0.3 to 0.3): {both_selective}")

# Colors from precomputed R, G, B columns
piYg_cmap = cm.get_cmap('PiYG_r')
selectivity = corr_df_filtered[['sp_mu_stack_idx']].values.ravel()
selectivity_norm = (selectivity + 1) / 2
selectivity_norm = np.clip(selectivity_norm, 0, 1)
colors = np.array([piYg_cmap(val)[:3] for val in selectivity_norm])

# Dot sizes scaled by magnitude
min_size = 1.5
max_size = 6
mag_min, mag_max = np.min(magnitude), np.max(magnitude)
magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min) if mag_max > mag_min else np.full_like(magnitude, 0.5)
sizes = np.clip(min_size + magnitude_norm * (max_size - min_size), min_size, max_size)
print(f"Electrode sizes range: {np.min(sizes):.2f} to {np.max(sizes):.2f}")

# SPECIAL ELECTRODE RENDER-ORDERING
tch30_ltg1_mask  = (corr_df_filtered['subj_id'] == 'TCH30') & (corr_df_filtered['channel'] == 132)
tch19_lstg6_mask = (corr_df_filtered['subj_id'] == 'TCH19') & (corr_df_filtered['channelnames'] == 'LSTG6')
s0022_mask       = (corr_df_filtered['subj_id'] == 'S0022') & (corr_df_filtered['channelnames'] == 'PST-PI6')
s0005_mask       = (corr_df_filtered['subj_id'] == 'S0005') & (corr_df_filtered['channelnames'] == 'RParGr3')

combined_special_mask    = tch30_ltg1_mask | tch19_lstg6_mask | s0022_mask | s0005_mask
special_indices_filtered = np.where(combined_special_mask.values)[0]
print(f"\nSpecial electrode indices: {special_indices_filtered}")
print(f"Number of special electrodes: {len(special_indices_filtered)}")

def reorder_to_plot_last(arr, indices_to_last):
    mask = np.ones(len(arr), dtype=bool)
    mask[indices_to_last] = False
    return np.concatenate([arr[mask], arr[indices_to_last]], axis=0)

def create_brain_mesh(surf):
    vertices = surf['vert']
    faces = surf['tri']
    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    return pv.PolyData(vertices, faces_pv)

def rotate_image(img, rotation_angle):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    return img.rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255, 0))

def rotate_final_image(image_path, rotation_angle):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255)).save(image_path)
    print(f"Rotated final image by {rotation_angle} degrees")

def create_and_save_view_hybrid(hemisphere, view_type, surf, elecs_subset, colors_subset, sizes_subset, magnitude_subset,
                                 camera_position, camera_focal_point, camera_up, tch30_indices=None):
    print(f"Creating {hemisphere} hemisphere {view_type} view")

    if tch30_indices is not None and len(tch30_indices) > 0:
        elecs_subset     = reorder_to_plot_last(elecs_subset, tch30_indices)
        colors_subset    = reorder_to_plot_last(colors_subset, tch30_indices)
        sizes_subset     = reorder_to_plot_last(sizes_subset, tch30_indices)
        magnitude_subset = reorder_to_plot_last(magnitude_subset, tch30_indices)

    window_size = [1200, 1200]

    plotter_brain = pv.Plotter(off_screen=True, window_size=window_size)
    plotter_brain.set_background([1.0, 1.0, 1.0])
    plotter_brain.add_mesh(create_brain_mesh(surf), color='lightgray', opacity=0.1, smooth_shading=True)
    plotter_brain.camera_position = [camera_position, camera_focal_point, camera_up]
    brain_screenshot_path = os.path.join(save_dir, f"brain_only_{hemisphere}_{view_type}.png")
    plotter_brain.screenshot(brain_screenshot_path, window_size=window_size, scale=3, transparent_background=False)
    plotter_brain.close()

    brain_img = Image.open(brain_screenshot_path)
    if 'lateral' in view_type:
        brain_img = rotate_image(brain_img, -90 if hemisphere == 'left' else 90)

    plotter_elecs = pv.Plotter(off_screen=True, window_size=[brain_img.width, brain_img.height])
    for i in range(len(elecs_subset)):
        opacity = 0.8 if magnitude_subset[i] >= 0.15 else 0.1
        sphere = pv.Sphere(radius=sizes_subset[i]/2, center=elecs_subset[i], theta_resolution=50, phi_resolution=50)
        plotter_elecs.add_mesh(sphere, color=colors_subset[i], opacity=opacity, lighting=False)
    plotter_elecs.camera_position = [camera_position, camera_focal_point, camera_up]
    elecs_screenshot_path = os.path.join(save_dir, f"elecs_only_{hemisphere}_{view_type}.png")
    plotter_elecs.screenshot(elecs_screenshot_path, window_size=[brain_img.width, brain_img.height], transparent_background=True)
    plotter_elecs.close()

    elecs_img = Image.open(elecs_screenshot_path)
    if 'lateral' in view_type:
        elecs_img = rotate_image(elecs_img, -90 if hemisphere == 'left' else 90)

    dpi = 300
    fig = plt.figure(figsize=(brain_img.width/dpi, brain_img.height/dpi), dpi=dpi, facecolor='white')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    ax.imshow(brain_img)
    ax.imshow(elecs_img)
    ax.axis('off')
    ax.set_xlim(0, brain_img.width)
    ax.set_ylim(brain_img.height, 0)

    output_path = os.path.join(save_dir, f"{hemisphere}_{view_type}_selectivity_{current_date}.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
    plt.close()

    if 'lateral' in view_type:
        rotate_final_image(output_path, 90 if hemisphere == 'left' else -90)

    os.remove(brain_screenshot_path)
    os.remove(elecs_screenshot_path)
    print(f"Successfully saved {output_path}")


def create_all_visible_view_hybrid(hemisphere, view_type, surf, elecs_subset, colors_subset, sizes_subset,
                                    camera_position, camera_focal_point, camera_up, tch30_indices=None):
    print(f"Creating ALL VISIBLE {hemisphere} hemisphere {view_type} view")

    if tch30_indices is not None and len(tch30_indices) > 0:
        elecs_subset  = reorder_to_plot_last(elecs_subset, tch30_indices)
        colors_subset = reorder_to_plot_last(colors_subset, tch30_indices)
        sizes_subset  = reorder_to_plot_last(sizes_subset, tch30_indices)

    window_size = [1200, 1200]

    plotter_brain = pv.Plotter(off_screen=True, window_size=window_size)
    plotter_brain.set_background([1.0, 1.0, 1.0])
    plotter_brain.add_mesh(create_brain_mesh(surf), color='lightgray', opacity=0.1, smooth_shading=True)
    plotter_brain.camera_position = [camera_position, camera_focal_point, camera_up]
    brain_screenshot_path = os.path.join(save_dir, f"brain_only_{hemisphere}_{view_type}.png")
    plotter_brain.screenshot(brain_screenshot_path, window_size=window_size, scale=3, transparent_background=False)
    plotter_brain.close()

    brain_img = Image.open(brain_screenshot_path)
    if 'lateral' in view_type:
        brain_img = rotate_image(brain_img, -90 if hemisphere == 'left' else 90)

    plotter_elecs = pv.Plotter(off_screen=True, window_size=[brain_img.width, brain_img.height])
    for i in range(len(elecs_subset)):
        sphere = pv.Sphere(radius=sizes_subset[i]/2, center=elecs_subset[i], theta_resolution=50, phi_resolution=50)
        plotter_elecs.add_mesh(sphere, color=colors_subset[i], opacity=1.0,
                               lighting=True, specular=0.0, ambient=0.4, diffuse=0.8)
    plotter_elecs.camera_position = [camera_position, camera_focal_point, camera_up]
    elecs_screenshot_path = os.path.join(save_dir, f"elecs_only_{hemisphere}_{view_type}_all.png")
    plotter_elecs.screenshot(elecs_screenshot_path, window_size=[brain_img.width, brain_img.height], transparent_background=True)
    plotter_elecs.close()

    elecs_img = Image.open(elecs_screenshot_path)
    if 'lateral' in view_type:
        elecs_img = rotate_image(elecs_img, -90 if hemisphere == 'left' else 90)

    dpi = 300
    fig = plt.figure(figsize=(brain_img.width/dpi, brain_img.height/dpi), dpi=dpi, facecolor='white')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    ax.imshow(brain_img)
    ax.imshow(elecs_img)
    ax.axis('off')
    ax.set_xlim(0, brain_img.width)
    ax.set_ylim(brain_img.height, 0)

    output_path = os.path.join(save_dir, f"{hemisphere}_{view_type}_selectivity_all_visible_{current_date}.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
    plt.close()

    if 'lateral' in view_type:
        rotate_final_image(output_path, 90 if hemisphere == 'left' else -90)

    os.remove(brain_screenshot_path)
    os.remove(elecs_screenshot_path)
    print(f"Successfully saved {output_path}")

# SPLIT BY HEMISPHERE 
left_mask  = elecs[:, 0] < 0
right_mask = elecs[:, 0] > 0

left_elecs     = elecs[left_mask]
left_colors    = colors[left_mask]
left_sizes     = sizes[left_mask]
left_magnitude = magnitude[left_mask]

right_elecs     = elecs[right_mask]
right_colors    = colors[right_mask]
right_sizes     = sizes[right_mask]
right_magnitude = magnitude[right_mask]

left_global_indices  = np.where(left_mask)[0]
right_global_indices = np.where(right_mask)[0]
left_special_local   = np.where(np.isin(left_global_indices,  special_indices_filtered))[0]
right_special_local  = np.where(np.isin(right_global_indices, special_indices_filtered))[0]

print(f"\nLeft hemisphere electrodes:  {sum(left_mask)}")
print(f"Right hemisphere electrodes: {sum(right_mask)}")
print(f"Special left  local indices: {left_special_local}")
print(f"Special right local indices: {right_special_local}")

left_lateral_cam  = ([-400, 0, 0], [0, 0, 0], [0, 0, 1])
right_lateral_cam = ([400,  0, 0], [0, 0, 0], [0, 0, 1])
left_dorsal_cam   = ([-100, 0, 600], [0, 0, 0], [0, 1, 0])
right_dorsal_cam  = ([100,  0, 600], [0, 0, 0], [0, 1, 0])

if sum(left_mask) > 0:
    print("\nProcessing Left Hemisphere - Lateral View")
    create_and_save_view_hybrid("left", "lateral", surf_lh,
                                left_elecs, left_colors, left_sizes, left_magnitude,
                                *left_lateral_cam, tch30_indices=left_special_local)

    print("Processing Left Hemisphere - All Visible - Lateral View")
    create_all_visible_view_hybrid("left", "lateral", surf_lh,
                                   left_elecs, left_colors, left_sizes,
                                   *left_lateral_cam, tch30_indices=left_special_local)

    print("Processing Left Hemisphere - Dorsal View")
    create_and_save_view_hybrid("left", "dorsal", surf_lh,
                                left_elecs, left_colors, left_sizes, left_magnitude,
                                *left_dorsal_cam, tch30_indices=left_special_local)

    print("Processing Left Hemisphere - All Visible - Dorsal View")
    create_all_visible_view_hybrid("left", "dorsal", surf_lh,
                                   left_elecs, left_colors, left_sizes,
                                   *left_dorsal_cam, tch30_indices=left_special_local)

if sum(right_mask) > 0:
    print("\nProcessing Right Hemisphere - Lateral View")
    create_and_save_view_hybrid("right", "lateral", surf_rh,
                                right_elecs, right_colors, right_sizes, right_magnitude,
                                *right_lateral_cam, tch30_indices=right_special_local)

    print("Processing Right Hemisphere - All Visible - Lateral View")
    create_all_visible_view_hybrid("right", "lateral", surf_rh,
                                   right_elecs, right_colors, right_sizes,
                                   *right_lateral_cam, tch30_indices=right_special_local)

    print("Processing Right Hemisphere - Dorsal View")
    create_and_save_view_hybrid("right", "dorsal", surf_rh,
                                right_elecs, right_colors, right_sizes, right_magnitude,
                                *right_dorsal_cam, tch30_indices=right_special_local)

    print("Processing Right Hemisphere - All Visible - Dorsal View")
    create_all_visible_view_hybrid("right", "dorsal", surf_rh,
                                   right_elecs, right_colors, right_sizes,
                                   *right_dorsal_cam, tch30_indices=right_special_local)

# Colorbar
print("\nGenerating colorbar legend...")
fig_colorbar = plt.figure(figsize=(8, 2), facecolor='white')
ax = fig_colorbar.add_axes([0.05, 0.5, 0.9, 0.4])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect='auto', cmap='PiYG_r')
ax.set_xticks([0, 128, 256])
ax.set_xticklabels(['Music Selective', 'Both', 'Speech Selective'])
ax.set_yticks([])
ax.set_xlabel('Selectivity Index (Variance Partitioning)', fontsize=12)
colorbar_path = os.path.join(save_dir, f"selectivity_colorbar_{current_date}.png")
plt.savefig(colorbar_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Colorbar saved to: {colorbar_path}")
plt.close()

print("\nProcessing complete!")