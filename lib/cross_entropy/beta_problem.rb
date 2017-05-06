# frozen_string_literal: true
module CrossEntropy
  #
  # Solve a continuous optimisation problem in which the variables are bounded
  # to the unit interval, [0, 1]. The sampling distribution of each parameter
  # is assumed to be a Beta distribution with parameters alpha and
  # beta.
  #
  class BetaProblem < AbstractProblem
    include NMath

    def initialize(alpha, beta)
      super [alpha, beta]

      to_generate_samples { generate_beta_samples }
      to_estimate { |elite| estimate_mom(elite) }

      yield(self) if block_given?
    end

    def param_alpha
      params[0]
    end

    def param_beta
      params[1]
    end

    #
    # Generate samples.
    #
    def generate_beta_samples
      NArray[*param_alpha.to_a.zip(param_beta.to_a).map do |alpha, beta|
        generate_beta_sample(alpha, beta)
      end]
    end

    #
    # Method of moments estimate using only the given 'elite' solutions.
    #
    # Maximum likelihood estimates for the parameters of the beta distribution
    # are difficult to compute, so we use the method of moments instead; see
    # http://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
    # for more information.
    #
    # @param [NArray] elite elite samples; dimension 0 is the sample index; the
    #        remaining dimensions contain the samples
    #
    # @return [Array] the estimated parameter arrays
    #
    def estimate_mom(elite)
      mean = elite.mean(0)
      variance = elite.stddev(0)**2

      q = mean * (1.0 - mean)
      valid = 0 < variance && variance < q
      r = q[valid] / variance[valid] - 1

      alpha = NArray[*param_alpha.map(&:to_f)]
      alpha[valid] = mean[valid] * r

      beta = NArray[*param_beta.map(&:to_f)]
      beta[valid] = (1.0 - mean[valid]) * r

      [alpha, beta]
    end

    private

    def generate_erlang_samples(k)
      -log(NArray.float(k, num_samples).random).sum(0)
    end

    def generate_beta_sample(alpha, beta)
      a = generate_erlang_samples(alpha)
      b = generate_erlang_samples(beta)
      a / (a + b)
    end
  end
end
