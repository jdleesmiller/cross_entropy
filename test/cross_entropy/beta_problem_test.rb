# frozen_string_literal: true
require_relative 'test_helper'

class TestBetaProblem < CrossEntropyTest
  #
  # Numerical tolerance for comparison. We would have to run for a long time to
  # get within the default tolerance of 10^-6, so use a less strict tolerance.
  #
  def delta
    1e-3
  end

  #
  # See http://en.wikipedia.org/wiki/Rosenbrock_function
  #
  # The function has a global minimum at $(a, a^2)$, but it's hard to find.
  #
  def test_rosenbrock_banana
    NArray.srand(567) # must use NArray's generator, not Ruby's

    a = 0.5
    b = 100.0
    smooth = 0.1

    alpha = NArray[1.0, 1.0]
    beta = NArray[1.0, 1.0]

    problem = CrossEntropy::BetaProblem.new(alpha, beta)
    problem.num_samples = 1000
    problem.num_elite   = 10
    problem.max_iters   = 10

    problem.to_score_sample { |x| (a - x[0])**2 + b * (x[1] - x[0]**2)**2 }

    problem.to_update do |new_alpha, new_beta|
      smooth_alpha = smooth * new_alpha + (1 - smooth) * problem.param_alpha
      smooth_beta = smooth * new_beta + (1 - smooth) * problem.param_beta
      [smooth_alpha, smooth_beta]
    end

    problem.solve

    estimates = problem.param_alpha / (problem.param_alpha + problem.param_beta)
    assert_narray_close NArray[0.5, 0.25], estimates
    assert problem.num_iters <= problem.max_iters
  end
end
