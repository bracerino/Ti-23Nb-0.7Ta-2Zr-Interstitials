# Ti-23Nb-0.7Ta-2Zr-Interstitials

This repository contains scripts related to the manuscript "Revealing interstitial energetics in Ti-23Nb-0.7Ta-2Zr gum metal base alloy via universal machine learning interatomic potentials"
All-in-one bash script ./SQS-ATAT/.SQS_Ti-23Nb-0_7Ta-2Zr.sh creates special quasirandom structures (SQSs) with 250-atom supercell approximating the composiiton of Ti-23Nb-0.7Ta-2Zr. The bash script was generated with our (SimplySQS)[https://simplysqs.com] application.
Folder ./uMLIPs-GO contains all-in-on Python scripts to perform structure optimizations with three universal machine-learning interatomic potentials (MACE-MATPES-PBE-0, Orb-v3, SevenNet-0) on *.poscar files in the sample directory as where the script will be placed. These scripts were generated with our (uMLIP-Interactive)[https://github.com/bracerino/uMLIP-Interactive] application.
