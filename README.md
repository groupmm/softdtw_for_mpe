# softdtw_for_mpe

This is a Pytorch code repository accompanying the following paper:  

```bibtex
@inproceedings{KrauseWM23_SoftDTWForMPE_ICASSP,
  author    = {Michael Krause and Christof Wei{\ss} and Meinard M{\"u}ller},
  title     = {Soft Dynamic Time Warping for Multi-Pitch Estimation and Beyond},
  booktitle = {Proceedings of the {IEEE} International Conference on Acoustics, Speech, and Signal Processing ({ICASSP})},
  address   = {Rhodes Island, Greece},
  doi       = {10.1109/ICASSP49357.2023.10095907},
  year      = {2023}
}
```

This repository contains code and trained models for paper's experiments. Some of the datasets used in the paper are partially available:

* [Schubert Winterreise Dataset (SWD)](https://zenodo.org/record/5139893#.YWRcktpBxaQ)
* [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)
* [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro)  

The codebase builds upon the [multipitch_mctc](https://github.com/christofw/multipitch_mctc) repository by Christof Weiß.
We further use the [CUDA implementation of SoftDTW](https://github.com/Maghoumi/pytorch-softdtw-cuda) by Mehran Maghumi.

For details and references, please see the paper.

# Getting Started
## Installation
```bash
cd softdtw_for_mpe
conda env create -f environment.yml
conda activate softdtw_for_mpe
```

## Data Preparation
1. Obtain and extract the datasets it in the ```data/``` subdirectory of this repository. 
2. Precompute inputs and targets:
```bash
python data_prep/01_extract_hcqt_pitch_schubert_winterreise.py
python data_prep/02_extract_overtone_target_schubert_winterreise.py
```
3. Extract ```data/Schubert_Winterreise/pitch_hs512_nonaligned.zip``` in that same directory.
For data preparation for other datasets than Schubert Winterreise, please see [multipitch_mctc](https://github.com/christofw/multipitch_mctc).

After precomputation, your data directory should contain at least the following:
```
├── data
    └── Schubert_Winterreise
        ├── 01_RawData
        │   └── audio_wav
        ├── 02_Annotations
        │   └── ann_audio_note
        ├── hcqt_hs512_o6_h5_s1
        ├── pitch_hs512_nooverl
        ├── pitch_hs512_overtones
        └── pitch_hs512_nonaligned
```
Here, ```01_RawData``` and ```02_Annotations``` originate from the SWD.
```hcqt_hs512_o6_h5_s1``` contains precomputed HCQT representations used as network input.
```pitch_hs512_nooverl``` contains strongly aligned pitch annotations.
```pitch_hs512_overtones``` contains strongly aligned pitch annotations with a simple overtone model applied (required for the experiments in Section 5.1 of the paper).
```pitch_hs512_nonaligned``` contains weakly aligned pitch annotations, based on  MIDI data.

# Experiments

In the [experiments](experiments) folder, all scripts for experiments from the paper can be found. The subfolder [models](experiments/models) contains trained models for all these experiments, and corresponding [log files](experiments/logs) are also provided. Please note that re-training requires a GPU as well as the pre-processed training data (see [Data Preparation](#data-preparation)).

Run scripts using, e.g., the following commands:  
```bash
export CUDA_VISIBLE_DEVICES=0
python experiments/mpe_schubert_softdtw_W2.py
```

- The numbers in Table 1 and Table 2 are obtained using the ```mpe_schubert_softdtw_*.py``` scripts, or found in reference [1].
- The results in Table 3 are obtained using the ```mpe_crossdataset_*.py``` scripts, or found in reference [1].
- The results from Section 5.1 are produced by the ```overtones_schubert_*.py``` scripts.
- For results and code on training with cross-version targets, we refer to our follow-up paper: Krause et al.: "Weakly Supervised Multi-Pitch Estimation Using Cross-Version Alignment", ISMIR 2023.

## Setup / Training / Evaluation 

All experiments are configured in the respective scripts. The following options are most important to our experiments:

- ```label_type```: which data to use as optimization target
  - ```'aligned'```: strong pitch annotations (binary, frame-wise aligned; used for the cross-entropy baseline and the SoftDTW_S variant of the loss)
  - ```'mctc_style'```: pitch annotations with removed duplicates (used for the SoftDTW_W1 variant of the loss)
  - ```'mctc_style_stretched'```: pitch annotations with removed duplicates, stretched to the length of the input sequence (used for the SoftDTW_W2 variant of the loss)
  - ```'nonaligned'```: pitch annotations with note lengths, but not aligned to the audio (used for the SoftDTW_W3 variant of the loss)
  - ```'nonaligned_stretched'```: pitch annotations with note lengths, stretched to the length of the input sequence, but not aligned to the audio (used for the SoftDTW_W4 variant of the loss)
  - ```'nonaligned_cqt'```: magnitude CQT representation of another version than the input excerpt (real-valued, used in Section 5.2)

- ```gamma```: The SoftDTW softness parameter
- ```enable_strongly_aligned_training```: Switching to standard, strongly-aligned training with cross-entropy or regression losses
- ```overtone_targets```: Used for the experiment presented in Section 5.1


The steps which should performed are configured by the flags ```do_train```, ```do_val```, ```do_test```. 

