## How to Run the Experiments
To execute the paper’s experiments, run:
```bash
bash do.sh
```

For the same model:
When modifying the training epoch or shift phase, update loadindex and shiftphase.
For a different model:
Modify the lines containing python justexp_FILE and forcexp_FILE, and update the expname accordingly.


## forcing_5p.py
To modify the observation points, adjust the variables x_obs and y_obs.


## findtargetfreq_tosh.py, plotphase_tosh.py, shift_tosh.py
To change the spatiotemporal analysis area, adjust parameters such as boxstart, spongebuf, and spinup.
Note on filterfreq: This variable is used in findtargetfreq_tosh.py to resolve issues with outlier values when detecting the target frequency. It generally does not need modification unless the main frequency of the Kármán vortex is being incorrectly identified—then it may be adjusted or removed.


## vardo.sh
This script runs post-paper experiments (e.g., experiments with varying observation shift phases) and follows a similar structure as do.sh.
To modify the shifting phase:
When you want to change from shiftphase1 to shiftphase2, update the corresponding parameters in the script.
To modify the method by which the shift phase changes, use varshift_tosh.py (for a single intermediate shift) or varshift2_tosh.py (for a gradual change from shiftphase1 to shiftphase2).


## Mask-Related Instructions
Changes related to updating the observation mask (i.e., modifications in the do.sh code) are located in the imsi folder.
When updating the mask for model runs, ensure that you reconfigure and run the PDE_CNN (or equivalent) accordingly.
