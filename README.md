# Ti-23Nb-0.7Ta-2Zr-Interstitials

This repository provides scripts for special quasirandom structure (SQS) generation and structural optimizations with universal machine-learning interatomic potentials (uMLIPs) used in the manuscript (arXiv, preprint) **[“Revealing interstitial energetics in Ti-23Nb-0.7Ta-2Zr gum metal base alloy via universal machine-learning interatomic potentials.](https://arxiv.org/abs/2512.05568)”**

## Repository Structure

### `SQS-ATAT/`

- **`SQS_Ti-23Nb-0_7Ta-2Zr.sh`**  
  All-in-one script to generates 250-atom SQSs representing the Ti-23Nb-0.7Ta-2Zr composition.  
  Created using our application **[SimplySQS](https://simplysqs.com)**.

### `uMLIPs-GO/`
Includes all-in-one Python scripts for structure optimization of `*.poscar` files (placed in the same directory as the Python script) using three uMLIPs:

- **MACE-MATPES-PBE-0**  
- **Orb-v3**  
- **SevenNet-0**  

These scripts were generated with our **[uMLIP-Interactive](https://github.com/bracerino/uMLIP-Interactive)**.

---
