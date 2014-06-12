module CrossEntropy
  #
  # Assuming that the data are probabilities in an NArray (say dim 1 or dim 2
  # for now). Rows (NArray dimension 1) must sum to one. Columns (NArray
  # dimension 0) represent the quantities to be optimized.
  #
  # Caller should set seed with NArray.srand before calling.
  #
  class MatrixProblem
    using NArrayExtensions

    def initialize
      # Defaults.
      @pr = nil
      @max_iters = nil
      @track_overall_min = false
      @overall_min_score = 1.0/0.0
      @overall_min_score_sample = nil

      # Configurable procs.
      @generate_samples = proc { self.generate_samples_directly }
      @score_sample     = proc {|sample| raise "no scoring function provided" }
      @estimate         = proc {|elite|  self.estimate_ml(elite) }
      @update           = proc {|pr_est| pr_est }
      @stop_decision    = proc {
        raise "no max_iters provided" unless self.max_iters
        self.num_iters >= self.max_iters
      }

      yield(self) if block_given?
    end

    def num_variables; @pr.shape[1] end
    def num_values;    @pr.shape[0] end

    attr_accessor :pr
    attr_accessor :num_samples
    attr_accessor :num_elite
    attr_accessor :max_iters

    def to_generate_samples &block; @generate_samples = block end

    def to_score_sample &block; @score_sample = block end

    def to_estimate &block; @estimate = block end

    def to_update &block; @update = block end

    def for_stop_decision &block; @stop_decision = block end

    attr_reader :num_iters
    attr_reader :min_score
    attr_reader :elite_score

    # Keep track of the best sample we've ever seen; if the scoring function is
    # deterministic, then this is a quantity of major interest.
    attr_reader :overall_min_score
    attr_reader :overall_min_score_sample
    attr_accessor :track_overall_min

    def solve
      @num_iters = 0

      begin
        @min_score   = nil
        @elite_score = nil

        samples = @generate_samples.call

        # Score each sample.
        scores = NArray.float(self.num_samples)
        for i in 0...self.num_samples
          sample_i = samples[i,true]
          score_i  = @score_sample.call(sample_i)

          # Keep track of best ever if requested.
          if track_overall_min && score_i < overall_min_score
            @overall_min_score        = score_i
            @overall_min_score_sample = sample_i
          end

          scores[i] = score_i
        end

        # Find elite quantile (gamma).
        scores_sorted = scores.sort
        @min_score   = scores_sorted[0]
        @elite_score = scores_sorted[self.num_elite-1]

        # Take all samples with scores below (or equal to) gamma; note that
        # there may be more than num_elite, due to ties.
        elite = samples[(scores <= elite_score).where, true]

        # Compute new probability estimates.
        pr_est = @estimate.call(elite)

        # Update main probability estimates (pr).
        self.pr = @update.call(pr_est)

        @num_iters += 1
      end until @stop_decision.call
    end

    #
    # Generate samples directly from the probabilities matrix {#pr}.
    #
    # If your problem is tightly constrained, you may want to provide a custom
    # sample generation routine that avoids infeasible solutions; see
    # {#to_generate_samples}.
    #
    def generate_samples_directly
      self.pr.tile(1,1,self.num_samples).sample_pmf_dim.transpose(1,0)
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
    def estimate_ml elite
      pr_est = NArray.float(self.num_values, self.num_variables)
      for i in 0...num_variables
        elite_i = elite[true,i]
        for j in 0...num_values
          pr_est[j,i] = elite_i.eq(j).count_true
        end
      end
      pr_est /= elite.shape[0]
      pr_est
    end

    #
    # Find most likely solution so far based on given probabilities.
    #
    # @param [NArray] pr probability matrix with {#num_variables} rows and
    # {#num_values} columns; if not specified, the current {#pr} matrix is used
    #
    # @return [Narray] column vector with {#num_variables} integer entries in
    # [0, {#num_values})
    #
    def most_likely_solution pr=self.pr
      pr_eq = pr.eq(pr.max(0).tile(1,pr.shape[0]).transpose(1,0))
      pr_ml = NArray.int(pr_eq.shape[1])
      for i in 0...pr_eq.shape[1]
        pr_ml[i] = pr_eq[true,i].where[0]
      end
      pr_ml
    end
  end
end

