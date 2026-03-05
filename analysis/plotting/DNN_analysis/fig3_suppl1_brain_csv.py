import os
os.environ["ETS_TOOLKIT"] = "null"
os.environ["QT_API"] = "pyqt5"

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from datetime import datetime
import pyvista as pv
from PIL import Image
from collections import defaultdict
from scipy.spatial import cKDTree  # for snapping

import warnings
warnings.filterwarnings('ignore')

print("Complete temporal lobe electrode plotting with OBJ meshes...")

# Paths
csv_file = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_withRGB_03_01_idx.csv'
temporal_mesh_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_imaging/cvs_avg35_inMNI152/Meshes'
base_output_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/brain/fig3_roi_brains'
today_datetime = datetime.today().strftime('%m_%d')
output_dir = os.path.join(base_output_dir, 'temporal_brain', today_datetime)
os.makedirs(output_dir, exist_ok=True)

# ROI definitions  +  which ROIs need the elevated camera
roi_mapping = {
    'HG':  { 'name': "Heschl's Gyrus",            'color': '#2a9d8f' },
    'PT':  { 'name': 'Planum Temporale',           'color': '#e9c46a' },
    'PP':  { 'name': 'Planum Polare',              'color': '#f4a261' },
    'STG': { 'name': 'Superior Temporal Gyrus',    'color': '#264653' },
    'STS': { 'name': 'Superior Temporal Sulcus',   'color': '#e76f51' },
    'MTG': { 'name': 'Middle Temporal Gyrus',      'color': '#a78bfa' },
}

# These ROIs sit on the superior / medial surface (inside the Sylvian fissure)
SUPERIOR_ROIS = {'HG', 'PT', 'PP'}

# Load OBJ meshes  +  build KD-trees for snapping
print("Loading temporal lobe OBJ meshes...")
lh_obj_path = os.path.join(temporal_mesh_dir, 'lh_temporal.obj')
rh_obj_path = os.path.join(temporal_mesh_dir, 'rh_temporal.obj')

surf_lh = pv.read(lh_obj_path)
surf_rh = pv.read(rh_obj_path)

print(f"  LH: {surf_lh.n_points} vertices, {surf_lh.n_faces} faces")
print(f"  RH: {surf_rh.n_points} vertices, {surf_rh.n_faces} faces")

# KD-trees built once — used for bulk snapping below
kdtree_lh = cKDTree(surf_lh.points)
kdtree_rh = cKDTree(surf_rh.points)
print("  KD-trees built for both hemispheres")

# Load electrode data  +  compute selectivity / magnitude / colours / sizes
print("\nLoading electrode data...")
df = pd.read_csv(csv_file)

speech_music_mixed = np.nan_to_num(df['speech_music_corrs_DNN'].values, nan=0.0)
speech_only        = np.nan_to_num(df['speech_only_corrs_DNN'].values,  nan=0.0)
music_only         = np.nan_to_num(df['music_only_corrs_DNN'].values,   nan=0.0)

selectivity = np.zeros(len(df))
magnitude   = np.zeros(len(df))

for i in range(len(df)):
    stacked     = np.nan_to_num(df['stacked_corrs_DNN'].values,     nan=0.0)
    speech_only = np.nan_to_num(df['speech_only_corrs_DNN'].values, nan=0.0)
    music_only  = np.nan_to_num(df['music_only_corrs_DNN'].values,  nan=0.0)
    magnitude = np.maximum(np.maximum(stacked, speech_only), music_only)
    selectivity = pd.to_numeric(df['sp_mu_stack_idx'], errors='coerce').fillna(0.0).values

print(f"Magnitude range: {magnitude.min():.3f} to {magnitude.max():.3f}")

# PiYG colours
piYg_cmap        = cm.get_cmap('PiYG_r')
selectivity_norm = np.clip((selectivity + 1) / 2, 0, 1)
piyg_colors      = np.array([piYg_cmap(val)[:3] for val in selectivity_norm])

# Marker sizes (radius in mm for the spheres)
min_size, max_size = 1.5, 6.0
mag_min, mag_max   = magnitude.min(), magnitude.max()
magnitude_norm     = ((magnitude - mag_min) / (mag_max - mag_min)
                      if mag_max > mag_min
                      else np.full_like(magnitude, 0.5))
sizes = np.clip(min_size + magnitude_norm * (max_size - min_size), min_size, max_size)

# Group electrodes by ROI & hemisphere  ──  then SNAP in bulk
print("\nGrouping electrodes by ROI and hemisphere...")

elec_by_roi        = defaultdict(lambda: defaultdict(list))
piyg_by_roi        = defaultdict(lambda: defaultdict(list))
roi_colors_by_roi  = defaultdict(lambda: defaultdict(list))
sizes_by_roi       = defaultdict(lambda: defaultdict(list))
magnitude_by_roi   = defaultdict(lambda: defaultdict(list))

# First pass: collect raw coords per ROI+hem (no snapping yet)
for idx, row in df.iterrows():
    roi = row['short_anat']
    if roi not in roi_mapping:
        continue
    x, y, z = row['x'], row['y'], row['z']
    hem = 'lh' if x < 0 else 'rh'

    elec_by_roi[roi][hem].append([x, y, z])
    piyg_by_roi[roi][hem].append(piyg_colors[idx])
    roi_colors_by_roi[roi][hem].append(roi_mapping[roi]['color'])
    sizes_by_roi[roi][hem].append(sizes[idx])
    magnitude_by_roi[roi][hem].append(magnitude[idx])

# Convert to arrays
for roi in elec_by_roi:
    for hem in elec_by_roi[roi]:
        elec_by_roi[roi][hem]       = np.array(elec_by_roi[roi][hem])
        piyg_by_roi[roi][hem]       = np.array(piyg_by_roi[roi][hem])
        sizes_by_roi[roi][hem]      = np.array(sizes_by_roi[roi][hem])
        magnitude_by_roi[roi][hem]  = np.array(magnitude_by_roi[roi][hem])

# Bulk snap per hemisphere 
# Collect every electrode for each hemisphere into one array, snap once,
# then write the snapped coords back into elec_by_roi.
for hem, tree in [('lh', kdtree_lh), ('rh', kdtree_rh)]:
    # Build ordered list of (roi, index-within-roi) so we can write back
    roi_order = []          # list of roi labels in the same order as big_coords
    big_coords = []
    for roi in elec_by_roi:
        if hem in elec_by_roi[roi] and len(elec_by_roi[roi][hem]) > 0:
            n = len(elec_by_roi[roi][hem])
            big_coords.append(elec_by_roi[roi][hem])   # already an ndarray
            roi_order.extend([roi] * n)

    if len(big_coords) == 0:
        continue

    big_coords = np.vstack(big_coords)                 # (M, 3)
    dists, indices = tree.query(big_coords)            # bulk nearest-neighbour
    snapped = tree.data[indices]                       # (M, 3) snapped coords

    print(f"  [{hem}] Snapped {len(big_coords)} electrodes  |  "
          f"max disp: {dists.max():.2f} mm  |  mean disp: {dists.mean():.2f} mm")

    # Write snapped coords back, ROI by ROI
    offset = 0
    for roi in elec_by_roi:
        if hem not in elec_by_roi[roi] or len(elec_by_roi[roi][hem]) == 0:
            continue
        n = len(elec_by_roi[roi][hem])
        elec_by_roi[roi][hem] = snapped[offset:offset + n]
        offset += n

# Print final counts
for roi in elec_by_roi:
    for hem in elec_by_roi[roi]:
        print(f"  {roi} {hem}: {len(elec_by_roi[roi][hem])} electrodes")

# Camera positions
# Lateral (original) — used for STG / STS / MTG and combined plots
left_lateral_cam  = ([-400, 0, 0],  [0, 0, 0], [0, 0, 1])
right_lateral_cam = ([400, 0, 0],   [0, 0, 0], [0, 0, 1])

# Oblique superior-lateral — used for HG / PT / PP individual plots.
# Camera is moved up (+z) and kept partially lateral so you see the top
# of the temporal lobe while retaining some lateral context.
# up vector stays [0, 0, 1] (same as lateral) so the image orientation
# doesn't rotate — only the viewing tilt changes.
left_superior_cam  = ([-250, 0, 300], [0, 0, 0], [0, 0, 1])
right_superior_cam = ([250, 0, 300],  [0, 0, 0], [0, 0, 1])

def get_camera(hem, roi_short=None):
    """Pick camera tuple based on hemisphere and (optional) ROI.

    roi_short=None          → combined plot  → lateral
    roi_short in SUPERIOR_ROIS → elevated oblique
    otherwise               → lateral
    """
    if roi_short is not None and roi_short in SUPERIOR_ROIS:
        return left_superior_cam if hem == 'lh' else right_superior_cam
    else:
        return left_lateral_cam  if hem == 'lh' else right_lateral_cam

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# Core rendering function (hybrid: PyVista brain + PyVista electrodes → PIL composite)
def create_and_save_view_hybrid(hemisphere, view_type, surf_mesh,
                                elecs_subset, colors_subset, sizes_subset,
                                camera_position, camera_focal_point, camera_up,
                                filename_base):
    """Render brain + electrodes with matching cameras, composite, save."""
    print(f"  Creating {hemisphere} {filename_base} {view_type}...")

    window_size = [1200, 1200]
    cam_tuple   = [camera_position, camera_focal_point, camera_up]

    # 1. Brain layer 
    plotter_brain = pv.Plotter(off_screen=True, window_size=window_size)
    plotter_brain.set_background([1.0, 1.0, 1.0])
    plotter_brain.add_mesh(surf_mesh, color='lightgray', opacity=0.1,
                           smooth_shading=True)
    plotter_brain.camera_position = cam_tuple

    brain_path = os.path.join(output_dir, "temp_brain.png")
    plotter_brain.screenshot(brain_path, window_size=window_size, scale=3,
                             transparent_background=False)
    plotter_brain.close()
    brain_img = Image.open(brain_path)

    # 2. Electrode layer 
    plotter_elecs = pv.Plotter(off_screen=True,
                               window_size=[brain_img.width, brain_img.height])

    for i in range(len(elecs_subset)):
        color = colors_subset[i]
        if isinstance(color, str):
            color = hex_to_rgb(color)

        sphere = pv.Sphere(radius=sizes_subset[i] / 2,
                           center=elecs_subset[i],
                           theta_resolution=50, phi_resolution=50)
        plotter_elecs.add_mesh(sphere, color=color, opacity=1.0,
                               lighting=True, specular=0.0,
                               ambient=0.4, diffuse=0.8)

    plotter_elecs.camera_position = cam_tuple

    elec_path = os.path.join(output_dir, "temp_elecs.png")
    plotter_elecs.screenshot(elec_path,
                             window_size=[brain_img.width, brain_img.height],
                             transparent_background=True)
    plotter_elecs.close()
    elec_img = Image.open(elec_path)

    # 3. Composite 
    dpi = 300
    fig = plt.figure(figsize=(brain_img.width / dpi, brain_img.height / dpi),
                     dpi=dpi, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    ax.imshow(brain_img)
    ax.imshow(elec_img)
    ax.axis('off')

    filename  = f"{filename_base}_{hemisphere}_{today_datetime}_{view_type}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0,
                facecolor='white', edgecolor='none')
    plt.close()

    # Clean up temp files
    os.remove(brain_path)
    os.remove(elec_path)
    print(f"    ✓ Saved: {filename}")

# Combined all-ROI plots  (lateral view for both hemispheres)
print("Generating combined plots (all ROIs)...")

all_left_elecs, all_left_piyg, all_left_roi_colors, all_left_sizes   = [], [], [], []
all_right_elecs, all_right_piyg, all_right_roi_colors, all_right_sizes = [], [], [], []

for roi in roi_mapping:
    if roi not in elec_by_roi:
        continue
    if 'lh' in elec_by_roi[roi] and len(elec_by_roi[roi]['lh']) > 0:
        all_left_elecs.append(elec_by_roi[roi]['lh'])
        all_left_piyg.append(piyg_by_roi[roi]['lh'])
        all_left_roi_colors.extend(roi_colors_by_roi[roi]['lh'])
        all_left_sizes.append(sizes_by_roi[roi]['lh'])
    if 'rh' in elec_by_roi[roi] and len(elec_by_roi[roi]['rh']) > 0:
        all_right_elecs.append(elec_by_roi[roi]['rh'])
        all_right_piyg.append(piyg_by_roi[roi]['rh'])
        all_right_roi_colors.extend(roi_colors_by_roi[roi]['rh'])
        all_right_sizes.append(sizes_by_roi[roi]['rh'])

if all_left_elecs:
    all_left_elecs  = np.vstack(all_left_elecs)
    all_left_piyg   = np.vstack(all_left_piyg)
    all_left_sizes  = np.concatenate(all_left_sizes)

    cam = get_camera('lh', roi_short=None)   # lateral
    create_and_save_view_hybrid("lh", "roi",  surf_lh, all_left_elecs,  all_left_roi_colors, all_left_sizes,  *cam, "temporal_lobe_all_ROIs")
    create_and_save_view_hybrid("lh", "piyg", surf_lh, all_left_elecs,  all_left_piyg,       all_left_sizes,  *cam, "temporal_lobe_all_ROIs")

if all_right_elecs:
    all_right_elecs = np.vstack(all_right_elecs)
    all_right_piyg  = np.vstack(all_right_piyg)
    all_right_sizes = np.concatenate(all_right_sizes)

    cam = get_camera('rh', roi_short=None)   # lateral
    create_and_save_view_hybrid("rh", "roi",  surf_rh, all_right_elecs, all_right_roi_colors, all_right_sizes, *cam, "temporal_lobe_all_ROIs")
    create_and_save_view_hybrid("rh", "piyg", surf_rh, all_right_elecs, all_right_piyg,       all_right_sizes, *cam, "temporal_lobe_all_ROIs")

# Individual ROI plots  (elevated for HG/PT/PP, lateral for STG/STS/MTG)
print("Generating individual ROI plots...")

for roi_short in roi_mapping:
    if roi_short not in elec_by_roi:
        print(f"\nNo electrodes found for {roi_short}, skipping...")
        continue

    print(f"\n{roi_short} ({roi_mapping[roi_short]['name']}):")

    for hem, surf in [('lh', surf_lh), ('rh', surf_rh)]:
        if hem not in elec_by_roi[roi_short] or len(elec_by_roi[roi_short][hem]) == 0:
            continue

        elecs      = elec_by_roi[roi_short][hem]
        piyg_cols  = piyg_by_roi[roi_short][hem]
        roi_cols   = roi_colors_by_roi[roi_short][hem]
        sz         = sizes_by_roi[roi_short][hem]

        print(f"  {hem.upper()}: {len(elecs)} electrodes")

        # ← camera selection: elevated for HG/PT/PP, lateral otherwise
        cam = get_camera(hem, roi_short=roi_short)

        create_and_save_view_hybrid(hem, "roi",  surf, elecs, roi_cols,  sz, *cam, roi_short)
        create_and_save_view_hybrid(hem, "piyg", surf, elecs, piyg_cols, sz, *cam, roi_short)

# Colorbar legend
print("Creating colorbar...")
fig = plt.figure(figsize=(8, 2), facecolor='white')
ax  = fig.add_axes([0.05, 0.5, 0.9, 0.4])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect='auto', cmap='PiYG_r')
ax.set_xticks([0, 128, 256])
ax.set_xticklabels(['Music Selective', 'Both', 'Speech Selective'])
ax.set_yticks([])
ax.set_xlabel('Selectivity Index', fontsize=12)
colorbar_path = os.path.join(output_dir, f'selectivity_colorbar_{today_datetime}.png')
plt.savefig(colorbar_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()


print(f"Output directory: {output_dir}")
print(f"\nCombined plots (all ROIs — lateral view):")
for hem in ('lh', 'rh'):
    for vt in ('roi', 'piyg'):
        print(f"  temporal_lobe_all_ROIs_{hem}_{today_datetime}_{vt}.png")
print(f"\nIndividual ROI plots:")
for roi in roi_mapping:
    view_note = "ELEVATED" if roi in SUPERIOR_ROIS else "lateral"
    print(f"  {roi} ({view_note}):")
    for hem in ('lh', 'rh'):
        for vt in ('roi', 'piyg'):
            print(f"    {roi}_{hem}_{today_datetime}_{vt}.png")
print(f"\nColorbar:")
print(f"  selectivity_colorbar_{today_datetime}.png")