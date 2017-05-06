# frozen_string_literal: true

require_relative 'test_helper'

class TestNArrayExtensions < CrossEntropyTest
  using CrossEntropy::NArrayExtensions

  def test_array_cumsum
    assert_equal NArray[], NArray[].cumsum
    assert_equal NArray[0], NArray[0].cumsum
    assert_equal NArray[1], NArray[1].cumsum
    assert_equal NArray[0, 1], NArray[0, 1].cumsum
    assert_equal NArray[1, 1], NArray[1, 0].cumsum
    assert_equal NArray[1, 2], NArray[1, 1].cumsum
    assert_equal NArray[1, 3, 6], NArray[1, 2, 3].cumsum
  end

  def test_narray_sample_pmf
    # Sample from vector.
    v = NArray.float(3).fill!(1)
    v /= v.sum
    assert_equal 0, v.sample_pmf_dim(0, NArray[0.0])
    assert_equal 0, v.sample_pmf_dim(0, NArray[0.333])
    assert_equal 1, v.sample_pmf_dim(0, NArray[0.334])
    assert_equal 1, v.sample_pmf_dim(0, NArray[0.666])
    assert_equal 2, v.sample_pmf_dim(0, NArray[0.667])
    assert_equal 2, v.sample_pmf_dim(0, NArray[0.999])

    # Sample from vector with sum < 1.
    v = NArray[0.5, 0.2, 0.2]
    assert_equal 0, v.sample_pmf_dim(0, NArray[0.0])
    assert_equal 1, v.sample_pmf_dim(0, NArray[0.5])
    assert_equal 2, v.sample_pmf_dim(0, NArray[0.89])
    assert_equal 2, v.sample_pmf_dim(0, NArray[0.91])

    # Zero at start won't be sampled.
    v = NArray[0.0, 0.5, 0.5]
    assert_equal 1, v.sample_pmf_dim(0, NArray[0.0])
    assert_equal 1, v.sample_pmf_dim(0, NArray[0.1])
    assert_equal 2, v.sample_pmf_dim(0, NArray[0.9])

    # If all entries are zero, we just choose the last one arbitrarily.
    v = NArray[0.0, 0.0, 0.0]
    assert_equal 2, v.sample_pmf_dim(0, NArray[0.9])

    # Sample from square matrix.
    m = NArray.float(3, 3).fill!(1)
    m /= 3
    assert_equal \
      NArray[0, 0, 0], m.sample_pmf_dim(0, NArray[[0.0], [0.0], [0.0]])
    assert_equal \
      NArray[1, 0, 0], m.sample_pmf_dim(0, NArray[[0.4], [0.0], [0.0]])
    assert_equal \
      NArray[1, 2, 0], m.sample_pmf_dim(0, NArray[[0.4], [0.7], [0.0]])
    assert_equal NArray[0, 0, 0], m.sample_pmf_dim(1, NArray[[0.0, 0.0, 0.0]])
    assert_equal NArray[1, 0, 0], m.sample_pmf_dim(1, NArray[[0.4, 0.0, 0.0]])
    assert_equal NArray[1, 2, 0], m.sample_pmf_dim(1, NArray[[0.4, 0.7, 0.0]])

    # Sample from non-square matrix.
    m = NArray.float(3, 2).fill!(1)
    m /= 3
    assert_equal NArray[0, 0], m.sample_pmf_dim(0, NArray[[0.0], [0.0]])
    assert_equal NArray[1, 0], m.sample_pmf_dim(0, NArray[[0.4], [0.0]])
    assert_equal NArray[1, 2], m.sample_pmf_dim(0, NArray[[0.4], [0.7]])

    m = m.transpose(1, 0)
    assert_equal NArray[0, 0], m.sample_pmf_dim(1, NArray[[0.0, 0.0]])
    assert_equal NArray[1, 0], m.sample_pmf_dim(1, NArray[[0.4, 0.0]])
    assert_equal NArray[1, 2], m.sample_pmf_dim(1, NArray[[0.4, 0.7]])

    # Sample from a 3D array.
    a = NArray.float(4, 3, 2).fill!(1)
    a /= 2
    sa = a.sample_pmf_dim(2)
    assert_equal 2, sa.dim
    assert_equal [0, 1], sa.to_a.flatten.uniq.sort
  end

  def test_narray_index_to_subscript
    assert_raises(IndexError) { NArray[].index_to_subscript(0) }

    assert_equal [0], NArray[0].index_to_subscript(0)

    assert_equal [0], NArray[0, 0].index_to_subscript(0)
    assert_equal [1], NArray[0, 0].index_to_subscript(1)

    assert_equal [0, 0], NArray[[0, 0]].index_to_subscript(0)
    assert_equal [1, 0], NArray[[0, 0]].index_to_subscript(1)
    assert_raises(IndexError) { NArray[[0, 0]].index_to_subscript(2) }
    assert_raises(IndexError) { NArray[[0, 0]].index_to_subscript(3) }
    assert_raises(IndexError) { NArray[[0, 0]].index_to_subscript(4) }

    a = NArray.int(2, 2).indgen!
    assert_equal [0, 0], a.index_to_subscript(0)
    assert_equal [1, 0], a.index_to_subscript(1)
    assert_equal [0, 1], a.index_to_subscript(2)
    assert_equal [1, 1], a.index_to_subscript(3)
    assert_raises(IndexError) { a.index_to_subscript(4) }

    a = NArray.int(2, 3).indgen!
    (0...2).each do |j|
      (0...3).each do |i|
        assert_equal [j, i], a.index_to_subscript(a[j, i])
      end
    end

    a = NArray.int(3, 2).indgen!
    (0...3).each do |j|
      (0...2).each do |i|
        assert_equal [j, i], a.index_to_subscript(a[j, i])
      end
    end

    a = NArray.int(3, 2, 4).indgen!
    (0...3).each do |j|
      (0...2).each do |i|
        (0...4).each do |h|
          assert_equal [j, i, h], a.index_to_subscript(a[j, i, h])
        end
      end
    end
  end

  def test_narray_sample
    assert_equal [0], NArray[1.0].sample_pmf

    assert_equal [0], NArray[0.5, 0.5].sample_pmf(NArray[0])
    assert_equal [0], NArray[0.5, 0.5].sample_pmf(NArray[0.49])
    assert_equal [1], NArray[0.5, 0.5].sample_pmf(NArray[0.5])
    assert_equal [1], NArray[0.5, 0.5].sample_pmf(NArray[1.0])

    a = NArray[[0.5, 0.5]]
    assert_equal [0, 0], a.sample_pmf(NArray[0])
    assert_equal [0, 0], a.sample_pmf(NArray[0.49])
    assert_equal [1, 0], a.sample_pmf(NArray[0.5])
    assert_equal [1, 0], a.sample_pmf(NArray[1.0])

    a = NArray[[0.2, 0], [0.3, 0.2]]
    assert_equal [0, 0], a.sample_pmf(NArray[0])
    assert_equal [0, 0], a.sample_pmf(NArray[0.19])
    assert_equal [0, 1], a.sample_pmf(NArray[0.2]) # note [1,0] has 0 mass
    assert_equal [1, 1], a.sample_pmf(NArray[0.5])
    assert_equal [1, 1], a.sample_pmf(NArray[0.51])

    a = NArray[[[0, 0.2], [0.2, 0.2]], [[0.1, 0.1], [0.1, 0.1]]]
    assert_equal [1, 0, 0], a.sample_pmf(NArray[0]) # note [0,0,0] has 0 mass
    assert_equal [1, 0, 0], a.sample_pmf(NArray[0.1])
    assert_equal [0, 1, 0], a.sample_pmf(NArray[0.21])
    assert_equal [1, 1, 0], a.sample_pmf(NArray[0.41])
    assert_equal [1, 1, 0], a.sample_pmf(NArray[0.59])
    assert_equal [0, 0, 1], a.sample_pmf(NArray[0.61])
    assert_equal [1, 0, 1], a.sample_pmf(NArray[0.71])
    assert_equal [0, 1, 1], a.sample_pmf(NArray[0.81])
    assert_equal [1, 1, 1], a.sample_pmf(NArray[0.91])
    assert_equal [1, 1, 1], a.sample_pmf(NArray[1.0])
  end

  def test_sample_pmf_examples
    a = NArray[[0.1, 0.2, 0.7],
               [0.3, 0.5, 0.2],
               [0.0, 0.2, 0.8],
               [0.7, 0.1, 0.2]]
    assert_equal [4], a.sample_pmf_dim(0).shape

    assert_equal \
      NArray[2, 1, 2, 0],
      a.cumsum(0).sample_cdf_dim(0, NArray[[0.5], [0.5], [0.5], [0.5]])

    a = NArray.float(3, 3, 3).fill!(1).div!(3 * 3 * 3)
    assert_equal 3, a.sample_pmf.size
  end
end
