# Entailment-Edits

Setting up the `EasyEdit` python environment on HPC with Mamba:

```bash
module load mamba
source activate data/grp_dmpowell/.mamba/envs/EasyEditShared
python -m ipykernel install --user --name=EasyEdit
```

Or, if you have trouble making it into a jupyter kernel, can just load it in vscode as a python environment, which also works.

```
source activate data/grp_dmpowell/.mamba/envs/EasyEditShared
```