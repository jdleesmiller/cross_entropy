# cross_entropy

https://github.com/jdleesmiller/cross_entropy 

## SYNOPSIS

Implementations of the [Cross Entropy Method](https://en.wikipedia.org/wiki/Cross-entropy_method) for several types of problems. Uses [NArray](http://masa16.github.io/narray/) for the numerics, to achieve reasonable performance.

### What is the Cross Entropy method?

It's basically like a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) without the biological stuff. Instead, it works on nice, pure probability distributions. You start by specifying a probability distribution for the optimal values, based on your initial guess. The CEM then
- generates samples based on that distribution,
- scores them according to the objective function, and
- uses the highest-scoring samples to update the parameters of the probability distribution, so it converges on an optimal value.

It has relatively few tunable parameters, and it automatically balances diversification and intensification. It is robust to noise in the objective function, so it is very useful for parameter tuning and simulation work.

### Supported problem types

- MatrixProblem: For discrete optimisation problems. Each variable can take one of a fixed number of states. The sampling distribution is a defined by a probability mass function for each variable. The term "matrix problem" is based on the idea that we can write the PMFs for each variable into the rows (NArray dimension 1) of a matrix. For example:
               value 1 | value 2
    variable 1     0.3 | 0.7
    variable 2     0.9 | 0.1

- ContinuousProblem: For continuous unbounded problems. The sampling
  distribution is a univariate Gaussian.

- BetaProblem: For continous bounded problems. The sampling distribution is a
  Beta distribution.

### Usage

For example, here is the [Rosenbrock banana function](http://en.wikipedia.org/wiki/Rosenbrock_function) and a custom smooth updater. The function has a global minimum at $(a, a^2)$, but it's hard to find.

    a = 1.0
    b = 100.0
    smooth = 0.1

    mean = NArray[0.0, 0.0]
    stddev = NArray[10.0, 10.0]

    problem = CrossEntropy::ContinuousProblem.new(mean, stddev)
    problem.num_samples = 1000
    problem.num_elite   = 10
    problem.max_iters   = 300

    problem.to_score_sample {|x| (a - x[0])**2 + b*(x[1] - x[0]**2)**2 }

    problem.to_update {|new_mean, new_stddev|
      smooth_mean = smooth*new_mean + (1 - smooth)*problem.param_mean
      smooth_stddev = smooth*new_stddev + (1 - smooth)*problem.param_stddev
      [smooth_mean, smooth_stddev]
    }

    problem.solve
    # problems.param_mean => NArray[1.0, 1.0]

== INSTALLATION

    gem install cross_entropy

== LICENSE

(The MIT License)

Copyright (c) 2015 John Lees-Miller

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

