Global sensitivity analysis

Code for paper:
Variable importance analysis of wind turbine extreme responses with Shapley value explanation

https://doi.org/10.1016/j.renene.2024.121049


### Environment:
conda env create -f environment.yml  
pip install -r requirements.txt


### How to use:
#### 1) theoretical case:
python main_theoretical.py gsa_theoretical
#### 2) wind turbine response case
python main.py train_metamodels

python main.py uncertainty_quantification

python main.py feature_importance

python main.py shapley_iec  
-- results saved:  
    results/sh_iec_bTD.txt  
    results/sh_iec_Mx_blade.txt  
    results/sh_iec_Mx_tower.txt  
-- plots generated:  
    figures/shapley_iec_bTD.pdf  
    figures/shapley_iec_Mx_blade.pdf  
    figures/shapley_iec_Mx_tower.pdf

python main.py shapley_nataf  
-- results saved:  
    results/sh_nataf_bTD.txt  
    results/sh_nataf_Mx_blade.txt  
    results/sh_nataf_Mx_tower.txt  
-- plots generated:  
    figures/shapley_nataf_bTD.pdf  
    figures/shapley_nataf_Mx_blade.pdf  
    figures/shapley_nataf_Mx_tower.pdf

### To do:
Test the newly generated models
Should validate functions/train_metamodels.py