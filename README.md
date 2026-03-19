**Code for Agravat et al., 2026: _Human auditory cortex preferentially tracks speech over music without explicit attention_**

**Overview:**
This repository contains code for preprocessing intracranial sEEG recordings, fitting spectrotemporal receptive field (STRF) encoding models, and generating all figures and statistical analyses. Participants (ages 4–21, n=54) listened passively to naturalistic movie trailer clips containing overlapping speech and music. Audio was post-hoc separated into isolated speech and music streams using deep neural networks (Moises). STRF encoding models were fit to predict high-gamma neural activity (70–150 Hz) from four conditions: mixed, speech-separated, music-separated, and stacked (speech + music).

**Usage:**  
  1. Preprocess neural data and create h5 files (preproc/preproc_og/ECoG_create_h5_functions_mixed.py; preproc/ECoG_create_h5_functions_speechmusic.py)
  2. Fit STRF encoding models (analysis/fit_strfs/fit_STRF_mixed.py; fit_STRF_speechmusic.py; fit_STRF_stacked.py)
  3. Aggregate results into CSV (analysis/plotting/DNN_analysis/make_allmodels_csv.py)
  4. Run statistical analyses (analysis/stats/fig4.R; musical_training_LMER.R; musical_training_Mann_Whitney_U_Test.R)
  5. Generate figures (analysis/plotting/DNN_analysis/fig1A.py; fig1B.py... etc.)

**Data:**
  Preprocessed neural data and encoding model outputs are available upon reasonable request. 
  Audio stimuli (movie trailers) are commercially licensed and cannot be redistributed. Separated speech and music streams were derived post-hoc using Moises.

**Dependencies:**
  Python (3.8.3):
  numpy (1.24.4), scipy (1.10.1), pandas (2.0.3), matplotlib (3.7.5), h5py (3.10.0),
  librosa (audio processing; 0.10.2),
  pyvista (0.44.2), nibabel (brain surface visualization; 5.2.1),
  mplcursors (interactive scatter plots; 0.6). 
  R (4.4.1):
  lme4 (1.1.38), lmerTest (linear mixed-effects models; 3.2.0),
  ggplot2 (4.0.2), ggeffects (2.3.2), patchwork (visualization; 1.3.2),
  readxl (1.4.5), dplyr (1.2.0), tidyr (data wrangling; 1.3.2). 
  Ridge regression utilities (ridge_.py, utils.py) are from the MNE-Python-based Hamilton Lab ridge regression toolbox.

