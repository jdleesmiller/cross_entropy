# frozen_string_literal: true

if ENV['COVERAGE']
  require 'simplecov'
  SimpleCov.start do
    add_filter '/test/'
  end
end

require 'cross_entropy'
require 'minitest/autorun'

class CrossEntropyTest < MiniTest::Test
  #
  # Tolerance for floating point comparison. Subclasses can override if they
  # need a different tolerance.
  #
  def delta
    1e-6
  end

  def assert_narray_close(exp, obs)
    assert exp.shape == obs.shape && ((exp - obs).abs < delta).all?,
           "#{exp.inspect} expected; got\n#{obs.inspect}"
  end
end
