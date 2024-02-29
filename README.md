# Entailment-Edits

To load the `EasyEdit` python environment on HPC with Mamba, run the following lines in the terminal on the cluster:

```bash
module load mamba
source activate /data/grp_dmpowell/.mamba/envs/EasyEditShared
```

You should then be able to load it as your environment for any script or notebook in vscode.

If you want it to appear as a jupyter kernel (for jupyterlab):

```bash
python -m ipykernel install --user --name=EasyEdit
```

But I had some issues with this and I believe it is not necessary.

## Config setup

Create a config.ini file with the following format:

```
[hugging_face]
token=your_token_here
[user]
username=your_username_here
```

The `.gitignore` should ensure this file is never committed to the repo. ***DO NOT ever let this be added to the repo!**

## Using Llama-2

The `playground.ipynb` notebook has a template for interacting with Llama-2. You can use that as a starting point.