# kg-rl-rts
Repo for researching/experimenting with KGs in RL using RTS environment (master thesis).

## Structure
- `kg-rl-rts` - main repo with experiments analysis, scripts, etc.
- `MicroRTS-Py` - submodule with Python game environment, contains inside another submodule `/gym_microrts/microrts` with Java game engine, both include KG modifications
- `results` - directory with results analysis, `/final_results.ipynb` - generation of thesis plots
- `final_results` - directory with result files of experiments included in the thesis
- `experiments` - directory with result files of old, preliminary experiments

Each KG experiment directory contains `graph.ttl` file with KG used for this particular run. Every experiment in `final_results` has exactly the same KG.
