## Facebook Prophet

After a failed installation it seems **Python < 3.9** is required unfortunately and anaconda is the recommended installation path on windows.

Not ideal, but I'll quickly set-up a conda environment to test this model.


```
    conda create -n dengue python=3.7.9
    conda install -c conda-forge prophet seaborn sckit-learn
    conda install jupyter ipykernel plotly

```



### Propeht Background
```
Prophet:
	y_t = f(t) (trend / curve fitting)

```

### Prophet does the following linear decomposition:
```
	g(t): Logistic or linear growth trend with optional linear splines (linear in the exponent for the logistic growth). The library calls the knots “change points.”
	s(t): Sine and cosine (i.e. Fourier series) for seasonal terms.
	h(t): Gaussian functions (bell curves) for holiday effects (instead of dummies, to make the effect smoother).
```




### Other things to Try:
1. ARIMA/VAR
2. ARIMAX/SARIMAX
3. ForecastHybrid (R) - mixture ETS/Auto.Arima

### Deep Learning Techniques:
1. Long ShortTerm Memory
2. **Recurrent Neural Net**
3. PyTorch