# Repairbads
Repairbads is an automatic and adaptive method to repair bad channels and segments for OPM-MEG.

Implementation code for the Repairbads method is provided here.

# Quick start
Using the Repairbads code requires raw continuous data with an MNE-Python raw structure and leadfield matrix.

To learn how to construct the raw structure, please refer [The Raw data structure: continuous data](https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html#the-raw-data-structure-continuous-data)

To obtain the lead field matrix, please refer to the following steps：
- Using fwd = [mne.make_forward_solution(...)](https://mne.tools/stable/generated/mne.make_forward_solution.html#mne.make_forward_solution) function to calculate a forward solution for a subject.
- The lead field matrix can be obtained using LF = fwd["sol"]["data"].

```python
from wfl_preproc_repairbads import repairbads
raw = mne.io.read_raw_fif(raw_path)
# LF: leadfield matrix
raw_repairbads = repairbads(raw,LF)
```

# Cite
If you use any part of the code, please cite the following publications:

- Wang F., Ma Y., Gao T., et al. Repairbads: An Automatic and Adaptive Method to Repair Bad Channels and Segments for OPM-MEG[J]. NeuroImage, 2025,306: 120996. [DOI:https://doi.org/10.1016/j.neuroimage.2024.120996](https://doi.org/10.1016/j.neuroimage.2024.120996)

# Acknowledgements

- [1] Mutanen T. P., Metsomaa J., Liljander S., et al. Automatic and Robust Noise Suppression in EEG and MEG: The SOUND Algorithm[J]. NeuroImage, 2018,166: 135-151.
- [2] De Cheveigné A., Parra L. C. Joint Decorrelation, a Versatile Tool for Multichannel Data Analysis[J]. NeuroImage, 2014,98: 487-505.

