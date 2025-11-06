# It's not that hard! Inferring problem difficulty with Large Language Models
This repository contains the code to 'It's not that hard! Inferring problem difficulty with Large Language Models' by Marthe Ballon, Andres Algaba, Brecht Verbeken and Vincent Ginis (arXiv link). 

## System requirements
```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python experiments/01_cmcqrd_create_pairs.py
```

## Overview of the data
1. The original three datasets used in the paper are available at
   - JEE Advanced Maths 2024 (https://jeeadv.ac.in/archive.html) 
   - The Cambridge MCQ Reading Dataset by Mullooly et al. 2023 (https://englishlanguageitutoring.com/datasets/cambridge-multiple-choice-questions-reading-dataset)
   - Omni-Math by Gao et al. 2024 (https://huggingface.co/datasets/KbsdJames/Omni-MATH)
  
2. The cleaned Omni-Math dataset is available at (...)
  
3. All data necessary to replicate the experiments is available at (10.5281/zenodo.17523641)

data/

&nbsp;&nbsp;&nbsp;&nbsp; jee/
&nbsp;&nbsp;&nbsp;&nbsp; - jee.parquet
- jee_pairs.parquet

cmcqrd/
- cmcqrd.parquet 
- cmcqrd_pairs.parquet
  
   omni/
   - omni.parquet (the subset of algebra questions from Omni-Math (cleaned) that do not contain any proofs, estimations or images)
   - omni_pairs.parquet

  
results/

   cmcqrd/
   - cmcqrd_with_bt.parquet
   - cmcqrd_with_labels.parquet
   - all_bt_with_difficulty_cmcqrd_oss.parquet


   omni/
   - omni_with_bt.parquet
   - omni_with_labels
   - omni_with_performance.parquet
   - all_bt_with_difficulty_omni_oss.parquet
   - omni_subsample_o3_correlations.parquet
   - omni_subsample_gemini_correlations.parquet

   - omni_o3_noise_alpha_0.01.parquet
   - omni_o3_noise_alpha_0.02.parquet
   - omni_o3_noise_alpha_0.05.parquet
   - omni_o3_noise_alpha_0.1.parquet

   - omni_gemini_noise_alpha_0.01.parquet
   - omni_gemini_noise_alpha_0.02.parquet
   - omni_gemini_noise_alpha_0.05.parquet
   - omni_gemini_noise_alpha_0.1.parquet

## Instructions for the data
Download the datafiles at the links provided above and insert them into the data/results folder indicated above. 


## Overview of the code

experiments/
- 01_cmcqrd_create_pairs.py
- 01_jee_create_pairs.py
- 01_omni_create_pairs.py

- 02_cmcqrd_batch_pairs.py
- 02_jee_batch_pairs.py
- 02_omni_batch_pairs.py

- 03_cmcqrd_process_results.py
- 03_jee_process_results.py
- 03_omni_process_results.py

- 04_cmcqrd_compute_bt.py
- 04_jee_compute_bt.py
- 04_omni_compute_bt.py

- 05_omni_benchmark.py

- 06_cmcqrd_label_by_llm.py
- 06_omni_label_by_llm.py

- 07_omni_add_noise_gemini.py
- 07_omni_add_noise_o3.py

- 08_omni_subsample.py

src/
- figures_main.ipynb
- figures_appendix.ipynb
    
- bt.py
- pairs.py
- prompts.py
- utils.py


## Instructions for the code
A detailed description on how to create the figures is provided in figures_main.ipynb and figures_appendix.ipynb. The python scripts 01-04 in experiments/ compute BT scores with OpenAI o3 and Gemini 2.5 Pro, for all three datasets. The scripts 05-08 create additional data to support the figures.





