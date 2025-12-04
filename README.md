# Ti-23Nb-0.7Ta-2Zr-Interstitials

This repository provides scripts used in the manuscript **“Revealing interstitial energetics in Ti-23Nb-0.7Ta-2Zr gum metal base alloy via universal machine-learning interatomic potentials.”**

## Repository Structure

### `SQS-ATAT/`

- **`SQS_Ti-23Nb-0_7Ta-2Zr.sh`**  
  All-in-one script to generates 250-atom special quasirandom structures (SQS) representing the Ti-23Nb-0.7Ta-2Zr composition.  
  Created using our application **[SimplySQS](https://simplysqs.com)**.

### `uMLIPs-GO/`
Includes all-in-one Python scripts for structure optimization of `*.poscar` files (placed in the same directory as the Python script) using three universal ML interatomic potentials:

- **MACE-MATPES-PBE-0**  
- **Orb-v3**  
- **SevenNet-0**  

These scripts were generated with our **[uMLIP-Interactive](https://github.com/bracerino/uMLIP-Interactive)**.

---
