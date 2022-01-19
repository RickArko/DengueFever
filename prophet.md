## Facebook Prophet

After a failed installation it seems **Python < 3.9** is required unfortunately and anaconda is the recommended installation path on windows.

Not ideal, but I'll quickly set-up a conda environment to test this model.


```
    conda create -n prophet python=3.8
    conda install -c conda-forge prophet
```