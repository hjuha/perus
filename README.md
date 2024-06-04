# Estimator for the Permanent of a Matrix

## Input

The program reads input from standard input stream in the following format:

```
n epsilon delta time_limit
a_11 a_12 ... a_1n
a_21 a_22 ... a_2n
...
a_n1 a_n2 ... a_nn
```

## Output

The problem outputs an approximation of the permanent of `A` with relative error at most `epsilon` with probability at least `1 - delta`.
