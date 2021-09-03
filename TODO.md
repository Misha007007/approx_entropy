# TODO

## Variance versus sample size

- Study the variance of the naive entropy estimator in examples to check empirically that the relationship is quadratic and not linear with respect to 1/n.
  - Is it $1/n$ or $1/n^2$? (in say, uniform distribution)
  - What is the undersampled regime in practice?
    - We say $n = \sqrt{s}$, but... in reality?

### Middle sample regime?

Between $\sqrt{s}$ and $s$, there is middle ground.

According to the model of variance, you can have different estimators:

- Constant, -> `DirectEstimator`
- $1/n^2$ -> `Estimator`. 
- Is there middle ground?

## Other distributions

For now, we have tested only Uniform distribution.

! We might have to code Ising distribution.

## Other estimators

How does this compare to:

- NSB
- Pillow
- Singleton
- Best upper bound
