# Thesis

## Usage
To run the external experiments, their folder first needs to be added to the `PYTHONPATH`:
#### Windows
```powershell
$env:PYTHONPATH = "external"
```
#### Linux/Unix
```shell
export PYTHONPATH=external:$PYTHONPATH
```
---
Once they are added, the "[Neural Networks and the Chomsky Hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main#)" experiments can be run using:
```shell
py .\experiments\main.py
```