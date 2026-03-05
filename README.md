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
  Python:
  numpy, scipy, pandas, matplotlib, h5py,
  librosa (audio processing),
  pyvista, nibabel (brain surface visualization),
  mplcursors (interactive scatter plots)
  
  R:
  lme4, lmerTest (linear mixed-effects models),
  ggplot2, ggeffects, patchwork (visualization),
  readxl, dplyr, tidyr (data wrangling). 

  Ridge regression utilities (ridge_.py, utils.py) are from the MNE-Python-based Hamilton Lab ridge regression toolbox.

