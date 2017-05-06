# frozen_string_literal: true

require_relative 'test_helper'

class TestCrossEntropy < CrossEntropyTest
  def test_ce_estimate_ml
    mp = CrossEntropy::MatrixProblem.new
    mp.params        = NArray.float(2, 4).fill!(0.5)
    mp.num_samples   = 50
    mp.num_elite     = 3

    # Note that the number of columns in elite can be > num_elite due to ties.
    elite = NArray[[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]
    pr_est = mp.estimate_ml(elite)
    assert_equal [2, 4], pr_est.shape
    assert_narray_close NArray[[0.75, 0.25],
                               [0.75, 0.25],
                               [0.75, 0.25],
                               [0.75, 0.25]], pr_est

    # All samples the same.
    elite = NArray[[0, 0, 0, 0],
                   [1, 1, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]
    pr_est = mp.estimate_ml(elite)
    assert_equal [2, 4], pr_est.shape
    assert_narray_close NArray[[1.0, 0.0],
                               [0.0, 1.0],
                               [1.0, 0.0],
                               [1.0, 0.0]], pr_est
  end

  def test_ce_most_likely_solution
    mp = CrossEntropy::MatrixProblem.new
    mp.params        = NArray.float(4, 3).fill!(0.25)
    mp.num_samples   = 50
    mp.num_elite     = 3

    # When there is a tie, the lowest value is taken.
    assert_equal NArray[0, 0, 0], mp.most_likely_solution

    mp.params = NArray[[0.0, 0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [0.2, 0.2, 0.2, 0.4]]
    assert_equal NArray[3, 0, 3], mp.most_likely_solution

    mp.params = NArray[[0.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.1, 0.3, 0.4, 0.2]]
    assert_equal NArray[2, 1, 2], mp.most_likely_solution
  end

  #
  # Example 1.2 from de Boer et al. 2005.
  # The aim is to search for the given Boolean vector y_true.
  # The MatrixProblem's default estimation rule is equivalent to equation (8).
  #
  def test_ce_deboer_1
    NArray.srand(567) # must use NArray's generator, not Ruby's

    n = 10
    y_true = NArray[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    mp = CrossEntropy::MatrixProblem.new
    mp.params        = NArray.float(2, n).fill!(0.5)
    mp.num_samples   = 50
    mp.num_elite     = 5
    mp.max_iters     = 10
    mp.track_overall_min = true

    mp.to_score_sample do |sample|
      y_true.eq(sample).count_false # to be minimized
    end

    mp.solve

    if y_true != mp.most_likely_solution
      warn "expected #{y_true}; found #{mp.most_likely_solution}"
    end

    if y_true != mp.overall_min_score_sample
      warn "expected overall #{y_true}; found #{mp.overall_min_score_sample}"
    end

    assert mp.num_iters <= mp.max_iters
  end

  #
  # Example 3.1 from de Boer et al. 2005.
  # This is a max-cut problem.
  # We also do some smoothing.
  #
  def test_ce_deboer_2
    NArray.srand(567) # must use NArray's generator, not Ruby's

    # Cost matrix
    n = 5
    c = NArray[[0, 1, 3, 5, 6],
               [1, 0, 3, 6, 5],
               [3, 3, 0, 2, 2],
               [5, 6, 2, 0, 2],
               [6, 5, 2, 2, 0]]

    mp = CrossEntropy::MatrixProblem.new
    mp.params = NArray.float(2, n).fill!(0.5)
    mp.params[true, 0] = NArray[0.0, 1.0] # put vertex 0 in subset 1
    mp.num_samples    = 50
    mp.num_elite      = 5
    mp.max_iters      = 10
    smooth            = 0.4

    max_cut_score = proc do |sample|
      weight = 0
      (0...n).each do |i|
        (0...n).each do |j|
          weight += c[j, i] if sample[i] < sample[j]
        end
      end
      -weight # to be minimized
    end
    best_cut = NArray[1, 1, 0, 0, 0]
    assert_equal(-15, max_cut_score.call(NArray[1, 0, 0, 0, 0]))
    assert_equal(-28, max_cut_score.call(best_cut))

    mp.to_score_sample(&max_cut_score)

    mp.to_update do |pr_iter|
      smooth * pr_iter + (1 - smooth) * mp.params
    end

    mp.for_stop_decision do
      # p mp.params
      mp.num_iters >= mp.max_iters
    end

    mp.solve

    if best_cut != mp.most_likely_solution
      warn "expected #{best_cut}; found #{mp.most_likely_solution}"
    end
    assert mp.num_iters <= mp.max_iters
  end
end
