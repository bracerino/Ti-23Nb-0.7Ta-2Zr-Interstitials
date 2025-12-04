# Ti-23Nb-0.7Ta-2Zr-Interstitials

This repository provides scripts used in the study **“Revealing interstitial energetics in Ti-23Nb-0.7Ta-2Zr gum metal base alloy via universal machine-learning interatomic potentials.”**

## Repository Structure

### `SQS-ATAT/`
Contains the all-in-one script:

- **`SQS_Ti-23Nb-0_7Ta-2Zr.sh`**  
  Generates 250-atom special quasirandom structures (SQS) representing the Ti-23Nb-0.7Ta-2Zr composition.  
  Created using **[SimplySQS](https://simplysqs.com)**.

### `uMLIPs-GO/`
Includes Python scripts for structure optimization of `*.poscar` files using three universal ML interatomic potentials:

- **MACE-MATPES-PBE-0**  
- **Orb-v3**  
- **SevenNet-0**  

These scripts were generated with **[uMLIP-Interactive](https://github.com/bracerino/uMLIP-Interactive)**.

---
