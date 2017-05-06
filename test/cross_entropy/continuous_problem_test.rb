# frozen_string_literal: true
require 'cross_entropy'
require 'minitest/autorun'

class TestContinuousProblem < MiniTest::Test
  # tolerance for numerical comparisons
  DELTA = 1e-6

  include NMath

  def assert_narray_close(exp, obs)
    assert exp.shape == obs.shape && ((exp - obs).abs < DELTA).all?,
           "#{exp.inspect} expected; got\n#{obs.inspect}"
  end

  #
  # Example 3.1 from Kroese et al. 2006.
  #
  # Maximise $e^{-(x-2)^2} + 0.8 e^{−(x+2)^2}$ for real $x$. The function has a
  # global maximum at x = 2 and a local maximum at x = -2, which we should
  # avoid.
  #
  # (This is also the example on Wikipedia.)
  #
  def test_Kroese_3_1
    NArray.srand(567) # must use NArray's generator, not Ruby's

    mean = NArray[0.0]
    stddev = NArray[10.0]

    problem = CrossEntropy::ContinuousProblem.new(mean, stddev)
    problem.num_samples = 100
    problem.num_elite   = 10
    problem.max_iters   = 100

    # NB: maximising
    problem.to_score_sample { |x| -(exp(-(x - 2)**2) + 0.8 * exp(-(x + 2)**2)) }

    problem.solve

    assert_narray_close NArray[2.0], problem.param_mean
    assert problem.num_iters <= problem.max_iters
  end

  #
  # See http://en.wikipedia.org/wiki/Rosenbrock_function
  #
  # The function has a global minimum at $(a, a^2)$, but it's hard to find.
  #
  def test_rosenbrock_banana
    NArray.srand(567) # must use NArray's generator, not Ruby's

    a = 1.0
    b = 100.0
    smooth = 0.1

    mean = NArray[0.0, 0.0]
    stddev = NArray[10.0, 10.0]

    problem = CrossEntropy::ContinuousProblem.new(mean, stddev)
    problem.num_samples = 1000
    problem.num_elite   = 10
    problem.max_iters   = 300

    problem.to_score_sample { |x| (a - x[0])**2 + b * (x[1] - x[0]**2)**2 }

    problem.to_update do |new_mean, new_stddev|
      smooth_mean = smooth * new_mean + (1 - smooth) * problem.param_mean
      smooth_stddev = smooth * new_stddev + (1 - smooth) * problem.param_stddev
      [smooth_mean, smooth_stddev]
    end

    problem.solve

    assert_narray_close NArray[1.0, 1.0], problem.param_mean
    assert problem.num_iters <= problem.max_iters
  end
end
