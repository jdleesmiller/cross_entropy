# frozen_string_literal: true
module CrossEntropy
  #
  # Solve a continuous optimisation problem. The sampling distribution of each
  # parameter is assumed to be a 1D Gaussian with given mean and variance.
  #
  class ContinuousProblem < AbstractProblem
    def initialize(mean, stddev)
      super [mean, stddev]

      to_generate_samples { generate_gaussian_samples }
      to_estimate { |elite| estimate_ml(elite) }

      yield(self) if block_given?
    end

    def param_mean
      params[0]
    end

    def param_stddev
      params[1]
    end

    def sample_shape
      param_mean.shape
    end

    #
    # Generate samples.
    #
    def generate_gaussian_samples
      r = NArray.float(num_samples, *sample_shape).randomn
      mean = param_mean.reshape(1, *sample_shape)
      stddev = param_stddev.reshape(1, *sample_shape)
      mean + stddev * r
    end

    #
    # Maximum likelihood estimate using only the given 'elite' solutions.
    #
    # @param [NArray] elite elite samples; dimension 0 is the sample index; the
    #        remaining dimensions contain the samples
    #
    # @return [Array] the estimated parameter arrays
    #
    def estimate_ml(elite)
      [elite.mean(0), elite.stddev(0)]
    end
  end
end
