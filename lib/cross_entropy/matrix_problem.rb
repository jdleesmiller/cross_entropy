# frozen_string_literal: true
module CrossEntropy
  #
  # Assuming that the data are probabilities in an NArray (say dim 1 or dim 2
  # for now). Rows (NArray dimension 1) must sum to one. Columns (NArray
  # dimension 0) represent the quantities to be optimized.
  #
  # Caller should set seed with NArray.srand before calling.
  #
  class MatrixProblem < AbstractProblem
    using NArrayExtensions

    def initialize(params = nil)
      super(params)

      to_generate_samples { generate_samples_directly }
      to_estimate { |elite| estimate_ml(elite) }

      yield(self) if block_given?
    end

    def num_variables
      @params.shape[1]
    end

    def num_values
      @params.shape[0]
    end

    #
    # Generate samples directly from the probabilities matrix {#params}.
    #
    # If your problem is tightly constrained, you may want to provide a custom
    # sample generation routine that avoids infeasible solutions; see
    # {#to_generate_samples}.
    #
    def generate_samples_directly
      params.tile(1, 1, num_samples).sample_pmf_dim.transpose(1, 0)
    end

    #
    # Maximum likelihood estimate using only the given 'elite' solutions.
    #
    # This is often (but not always) the optimal estimate for the probabilities
    # from the elite samples for problems of this form.
    #
    # @param [NArray] elite {#num_variables} rows; the number of columns depends
    # on the {#num_elite} parameter, but is typically less than {#num_samples};
    # elements are integer in [0, {#num_values})
    #
    # @return [NArray] {#num_variables} rows; {#num_values} columns; entries are
    # non-negative floats in [0,1] and sum to 1
    #
    def estimate_ml(elite)
      pr_est = NArray.float(num_values, num_variables)
      (0...num_variables).each do |i|
        elite_i = elite[true, i]
        (0...num_values).each do |j|
          pr_est[j, i] = elite_i.eq(j).count_true
        end
      end
      pr_est /= elite.shape[0]
      pr_est
    end

    #
    # Find most likely solution so far based on given probabilities.
    #
    # @param [NArray] pr probability matrix with {#num_variables} rows and
    # {#num_values} columns; if not specified, the current {#params} matrix is
    # used
    #
    # @return [Narray] column vector with {#num_variables} integer entries in
    # [0, {#num_values})
    #
    def most_likely_solution(pr = params)
      pr_eq = pr.eq(pr.max(0).tile(1, pr.shape[0]).transpose(1, 0))
      pr_ml = NArray.int(pr_eq.shape[1])
      (0...pr_eq.shape[1]).each do |i|
        pr_ml[i] = pr_eq[true, i].where[0]
      end
      pr_ml
    end
  end
end
