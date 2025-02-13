# PyPalettes
Repairbads is an automatic and adaptive method to repair bad channels and segments for OPM-MEG.

Implementation code for the Repairbads method is provided here.

# Quick start
```python
from wfl_preproc_repairbads import repairbads
raw = mne.io.read_raw_fif(raw_path)
# LF: leadfield matrix
raw_repairbads = wfl_preproc_repairbads(raw,LF)
```

If you use any part of the code, please cite the following publications:

Wang F., Ma Y., Gao T., et al. Repairbads: An Automatic and Adaptive Method to Repair Bad Channels and Segments for OPM-MEG[J]. NeuroImage, 2025,306: 120996.
