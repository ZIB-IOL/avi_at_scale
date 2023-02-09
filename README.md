# Approximate Vanishing Ideal Computations at Scale

Code for the paper:
Wirth, E., Kera, H., and Pokutta, S. (2023). Approximate Vanishing Ideal Computations at Scale. To Appear in Proceedings of ICLR.


## References
This project is an extension of the previously published Git Repository
[CGAVI](https://github.com/ZIB-IOL/cgavi/releases/tag/v1.0.0),
which is the code corresponding to the following paper:

[Wirth, E. S., & Pokutta, S. (2022, May). Conditional gradients for the approximately vanishing ideal. In Proceedings of the International Conference on Artificial Intelligence and Statistics (pp. 2191-2209). PMLR.](https://proceedings.mlr.press/v151/wirth22a.html)



## Installation guide
Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command: 
```shell script
$ conda env create --file environment.yml
```
This will create the conda environment avi_at_scale.

Activate the conda environment with:
```shell script
$ conda activate avi_at_scale
```

Run the tests:
```python3 script
>>> python3 -m unittest
```

No errors should occur.


Execute the experiments: 
```python3 script
>>> python3 experiments_all.py
```

This will create folders named data_frames and plots, which contain subfolders containing the experiment results and 
the plots, respectively. 

The performance experiments can be displayed as latex_code by executing:
```python3 script
>>> experiments_results_to_latex.py
```
