# Description

This directory has the code to generate weak 'Study type and design' labels for the EBM-PICO corpus. `label_generator.py` takes input EBM-PICO training set, weakly labels it using multiple labeling functions (LF) and saves the labeled candidates to local. Use `args` argument parser to set the program flow.

`level1` labels EBM-PICO input dataset with study type labels using concepts from UMLS ontologies for selected, study type mapped semantic types.
`level2` labels EBM-PICO input dataset with study type labels using non-UMLS ontologies all of which were downloaded from [NCBO BioPortal](https://bioportal.bioontology.org/).
`level3` labels EBM-PICO input dataset using several ReGeX's, heuristics and some hand-crafted dictionaries.

For the input sequence, a labelling function uses the abovementioned labelling sources and heuristics to output a sequence of labels. It labels a positive token with 1, a negative token with 0 and abstains (-1) wherever it cannot make a decision.