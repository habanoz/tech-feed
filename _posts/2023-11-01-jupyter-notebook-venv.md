---
layout: single
classes: wide
title:  "Running Jupyter Notebooks in Virtual Environments"
categories:
  - best-practices

tags:
  - bash
  - jupyter
---

As of 2023 November, jupyterlab still does not support working with virtual environments. 
Using global environment causes all sort of troubles. It is possible to add custom kernels to jupyterlab but it is a little bit involved. 

Finally, i decided to create bash aliases to simplify the process. Adding following code to `.bashrc` file allows to create a specific environment to work with jupyter lab.

The idea is to create a virtual environment, work on it and eventually remove it. In this flow, if notebook file should is necessary, it needs needs to backed up. This script can be customized to create different flows.


## Scripts

```bash

## alias definitions

function create_nb_env() {
    local python_version=${2:-3.8}
    conda create python=$python_version -p /venv/$1 -y > /dev/null 2>&1
    conda activate /venv/$1
    
    # create notebook kernel
    conda install -c anaconda ipykernel -y > /dev/null 2>&1
    python -m ipykernel install --user --name=$1
    pip install jupyterlab --quiet > /dev/null 2>&1
    
    # start notebook server
    mkdir /venv/$1/src
    jupyter lab --notebook-dir=/venv/$1/src
}
alias mknbenv=create_nb_env

function delete_nb_env() {
    jupyter kernelspec uninstall $1 -y
    conda deactivate
    conda env remove -p /venv/$1
}
alias rmnbenv=delete_nb_env
```


`/venv` is a sybolic link to any directory you want to keep virtual environments. I prefer '~/workspace/venv'. 

```bash
ln -s ~/workspace/venv /venv
sudo chmod 777 /venv
sudo chown /venv myuser
```


## Example usage:

```bash
mknbenv jupyter_env_1 3.10
rmnbenv jupyter_env_1
```