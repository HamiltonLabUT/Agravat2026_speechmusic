import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt

import pyvista as pv
# Set PyVista to work headless (no GUI window)
pv.set_plot_theme('document')
pv.global_theme.window_size = [2400, 2400]
pv.global_theme.background = 'white'
pv.OFF_SCREEN = True

from datetime import datetime

# Set save directory
save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/brain/dev_fig1a'
os.makedirs(save_dir, exist_ok=True)

# Get the current date to append to filenames
current_date = datetime.now().strftime('%m_%d')

# Load the electrode data
corr_df = pd.read_csv('/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05.csv')

# Filter out 'Unknown' completely
corr_df = corr_df[corr_df['short_anat'] != 'Unknown']

# Define the developmental stages to keep with their colors
dev_colors = {
    'Early childhood': '#98df8a',  # light green
    'Middle childhood': '#009c41',  # dark green
    'Early adolescence': '#3393ff',  # light blue
    'Late adolescence': '#021ca4'   # dark blue
}

# Separate electrodes: developmental stages in colored regions vs. other/supramar/insula in black
# First, get electrodes that should be black (specific anatomical regions)
other_filtered = corr_df[corr_df['short_anat'].isin(['other', 'supramar', 'insula'])]

# Then, get electrodes that have developmental stages AND are NOT in the black regions
dev_filtered = corr_df[
    (corr_df['dev'].isin(dev_colors.keys())) & 
    (~corr_df['short_anat'].isin(['other', 'supramar', 'insula']))
]

# Debug prints to check the filtering
print(f"Total electrodes after removing Unknown: {len(corr_df)}")
print(f"Developmental stage electrodes (colored): {len(dev_filtered)}")
print(f"Other/supramar/insula electrodes (black): {len(other_filtered)}")
print(f"Anatomical regions in dev_filtered: {dev_filtered['short_anat'].unique()}")
print(f"Anatomical regions in other_filtered: {other_filtered['short_anat'].unique()}")
print(f"Dev stages in dev_filtered: {dev_filtered['dev'].unique()}")
print(f"Overlap check - total plotted: {len(dev_filtered) + len(other_filtered)}")

# Function to load FreeSurfer surface files directly
def load_freesurfer_surface(surf_file):
    """
    Load FreeSurfer surface file directly
    Returns vertices and faces
    """
    try:
        import nibabel as nib
        # Try using nibabel if available
        surf = nib.freesurfer.read_geometry(surf_file)
        vertices = surf[0]
        faces = surf[1]
        return {'vert': vertices, 'tri': faces}
    except ImportError:
        print("nibabel not available, trying manual FreeSurfer reading...")
        # Manual FreeSurfer surface file reading
        return read_freesurfer_surface_manual(surf_file)

def read_freesurfer_surface_manual(surf_file):
    """
    Manual FreeSurfer surface file reader
    """
    with open(surf_file, 'rb') as f:
        # Read magic number
        magic = np.frombuffer(f.read(3), dtype=np.uint8)
        if not np.array_equal(magic, [255, 255, 254]):
            raise ValueError("Not a valid FreeSurfer surface file")
        
        # Skip one byte
        f.read(1)
        
        # Read comment
        comment = f.read(f.read(1)[0])
        
        # Read number of vertices and faces
        nvert = np.frombuffer(f.read(4), dtype='>i4')[0]
        nface = np.frombuffer(f.read(4), dtype='>i4')[0]
        
        # Read vertices
        vertices = np.frombuffer(f.read(nvert * 3 * 4), dtype='>f4').reshape((nvert, 3))
        
        # Read faces
        faces = np.frombuffer(f.read(nface * 3 * 4), dtype='>i4').reshape((nface, 3))
        
        return {'vert': vertices, 'tri': faces}

# Try to load brain surfaces
imaging_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_imaging'

# Paths to FreeSurfer surfaces (you may need to adjust these paths)
lh_surf_path = os.path.join(imaging_dir, 'cvs_avg35_inMNI152', 'surf', 'lh.pial')
rh_surf_path = os.path.join(imaging_dir, 'cvs_avg35_inMNI152', 'surf', 'rh.pial')

# Alternative surface paths to try
alternative_paths = [
    ('lh.inflated', 'rh.inflated'),
    ('lh.white', 'rh.white'),
    ('lh.smoothwm', 'rh.smoothwm')
]

surf_lh = None
surf_rh = None

# Try to load surfaces
for lh_name, rh_name in [('lh.pial', 'rh.pial')] + alternative_paths:
    lh_path = os.path.join(imaging_dir, 'cvs_avg35_inMNI152', 'surf', lh_name)
    rh_path = os.path.join(imaging_dir, 'cvs_avg35_inMNI152', 'surf', rh_name)
    
    try:
        if os.path.exists(lh_path) and os.path.exists(rh_path):
            print(f"Attempting to load {lh_name} and {rh_name}...")
            surf_lh = load_freesurfer_surface(lh_path)
            surf_rh = load_freesurfer_surface(rh_path)
            print(f"Successfully loaded {lh_name} and {rh_name}")
            break
    except Exception as e:
        print(f"Failed to load {lh_name}/{rh_name}: {e}")
        continue

# If surfaces couldn't be loaded, create simple brain-like meshes
if surf_lh is None or surf_rh is None:
    print("Could not load FreeSurfer surfaces, creating simple brain-like meshes...")
    # Create simple hemisphere meshes
    sphere_lh = pv.Sphere(radius=80, center=[-40, 0, 0], phi_resolution=50, theta_resolution=50)
    sphere_rh = pv.Sphere(radius=80, center=[40, 0, 0], phi_resolution=50, theta_resolution=50)
    
    # Convert to the format we expect
    surf_lh = {'vert': sphere_lh.points, 'tri': sphere_lh.faces.reshape(-1, 4)[:, 1:4]}
    surf_rh = {'vert': sphere_rh.points, 'tri': sphere_rh.faces.reshape(-1, 4)[:, 1:4]}

# Convert brain surfaces to PyVista meshes
def freesurfer_to_pyvista(surf):
    """Convert freesurfer surface to PyVista mesh"""
    points = surf['vert']
    faces = surf['tri']
    
    # PyVista expects faces as [n_points, point1, point2, point3, ...]
    n_faces = faces.shape[0]
    pv_faces = np.column_stack([np.full(n_faces, 3), faces]).flatten()
    
    mesh = pv.PolyData(points, pv_faces)
    return mesh

# Create PyVista meshes
mesh_lh = freesurfer_to_pyvista(surf_lh)
mesh_rh = freesurfer_to_pyvista(surf_rh)

print(f"Left hemisphere: {mesh_lh.n_points} vertices, {mesh_lh.n_faces} faces")
print(f"Right hemisphere: {mesh_rh.n_points} vertices, {mesh_rh.n_faces} faces")

# Prepare electrode data for developmental stages
elecs_dev = np.array((dev_filtered['x'], dev_filtered['y'], dev_filtered['z'])).T
dev_labels = dev_filtered['dev'].values
colors_dev = np.array([dev_colors[dev] for dev in dev_labels])

# Prepare electrode data for other anatomical regions (black, smaller)
elecs_other = np.array((other_filtered['x'], other_filtered['y'], other_filtered['z'])).T

def plot_brain_with_electrodes(meshes, elecs_dev, colors_dev, elecs_other, view_vector, up_vector, filename):
    """Plot brain with electrodes using PyVista"""
    plotter = pv.Plotter(off_screen=True, window_size=[2400, 2400])
    
    # Add brain surface(s)
    if isinstance(meshes, list):
        for mesh in meshes:
            plotter.add_mesh(mesh, color='lightgray', opacity=0.1, smooth_shading=True)
    else:
        plotter.add_mesh(meshes, color='lightgray', opacity=0.1, smooth_shading=True)
    
    # Add developmental stage electrodes (colored, larger)
    if len(elecs_dev) > 0:
        for i, (elec, color) in enumerate(zip(elecs_dev, colors_dev)):
            sphere = pv.Sphere(radius=1.5, center=elec)  # Larger spheres for dev stages
            plotter.add_mesh(sphere, color=color, smooth_shading=True)
    
    # Add other electrodes (black, smaller)
    if len(elecs_other) > 0:
        for elec in elecs_other:
            sphere = pv.Sphere(radius=0.5, center=elec)  # Much smaller spheres for others
            plotter.add_mesh(sphere, color='gray', smooth_shading=True)
    
    # Set camera position
    plotter.camera_position = [
        view_vector,  # camera position
        [0, 0, 0],    # focal point
        up_vector     # up vector
    ]
    
    # Save high-resolution images
    png_path = os.path.join(save_dir, f"{filename}_combined_{current_date}_pyvista.png")
    
    plotter.screenshot(png_path, window_size=[2400, 2400])
    print(f"Saved: {png_path}")
    
    plotter.close()

# Define camera positions for different views
views = {
    'left_lateral': ([-400, 0, 0], [0, 0, 1]),      # Left lateral view
    'right_lateral': ([400, 0, 0], [0, 0, 1]),      # Right lateral view
    'dorsal': ([0, 0, 450], [0, 1, 0]),             # Top view
    'ventral': ([0, 0, -450], [0, 1, 0]),           # Bottom view
}

# COMBINED PLOTS - BOTH HEMISPHERES 
for view_name, (view_vector, up_vector) in views.items():
    print(f"Creating {view_name} view...")
    plot_brain_with_electrodes([mesh_lh, mesh_rh], elecs_dev, colors_dev, elecs_other, 
                              view_vector, up_vector, view_name)

# LEFT HEMISPHERE PLOTS 
# Get left hemisphere electrodes (x < 0)
if len(elecs_dev) > 0:
    left_mask_dev = elecs_dev[:, 0] < 0
    elecs_lh_dev = elecs_dev[left_mask_dev]
    colors_lh_dev = colors_dev[left_mask_dev]
else:
    elecs_lh_dev = np.array([]).reshape(0, 3)
    colors_lh_dev = np.array([])

if len(elecs_other) > 0:
    left_mask_other = elecs_other[:, 0] < 0
    elecs_lh_other = elecs_other[left_mask_other]
else:
    elecs_lh_other = np.array([]).reshape(0, 3)

print("Creating left hemisphere views...")
plot_brain_with_electrodes(mesh_lh, elecs_lh_dev, colors_lh_dev, elecs_lh_other, 
                          [-400, 0, 0], [0, 0, 1], 'left_lateral_only')
plot_brain_with_electrodes(mesh_lh, elecs_lh_dev, colors_lh_dev, elecs_lh_other, 
                          [-200, 0, 300], [0, 1, 0], 'left_dorsal')

# RIGHT HEMISPHERE PLOTS 
# Get right hemisphere electrodes (x > 0)
if len(elecs_dev) > 0:
    right_mask_dev = elecs_dev[:, 0] > 0
    elecs_rh_dev = elecs_dev[right_mask_dev]
    colors_rh_dev = colors_dev[right_mask_dev]
else:
    elecs_rh_dev = np.array([]).reshape(0, 3)
    colors_rh_dev = np.array([])

if len(elecs_other) > 0:
    right_mask_other = elecs_other[:, 0] > 0
    elecs_rh_other = elecs_other[right_mask_other]
else:
    elecs_rh_other = np.array([]).reshape(0, 3)

print("Creating right hemisphere views...")
plot_brain_with_electrodes(mesh_rh, elecs_rh_dev, colors_rh_dev, elecs_rh_other, 
                          [400, 0, 0], [0, 0, 1], 'right_lateral_only')
plot_brain_with_electrodes(mesh_rh, elecs_rh_dev, colors_rh_dev, elecs_rh_other, 
                          [200, 0, 300], [0, 1, 0], 'right_dorsal')

# Create a legend figure using matplotlib
def plot_legend():
    plt.figure(figsize=(8, 5))
    # Plot developmental stage colors
    for i, (dev_stage, color) in enumerate(dev_colors.items()):
        plt.plot([0, 1], [1 - 0.15*i, 1 - 0.15*i], color=color, lw=6)
        plt.text(1.05, 1 - 0.15*i, dev_stage, va='center', fontsize=14)
    
    # Add black electrodes to legend
    plt.plot([0, 1], [1 - 0.15*len(dev_colors), 1 - 0.15*len(dev_colors)], color='black', lw=4)
    plt.text(1.05, 1 - 0.15*len(dev_colors), 'Other/Supramar/Insula', va='center', fontsize=14)
    
    plt.axis('off')
    legend_path = os.path.join(save_dir, f"dev_stage_legend_combined_{current_date}_pyvista.png")
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved legend: {legend_path}")

# Save legend figure
plot_legend()

print(f"\nPyVista plots saved to {save_dir}")
print(f"Total electrodes plotted: {len(dev_filtered) + len(other_filtered)}")