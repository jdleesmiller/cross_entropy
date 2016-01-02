module CrossEntropy
  #
  # Base class for specific problem types.
  #
  class AbstractProblem
    #
    # @param [Array] params
    #
    def initialize params
      @params = params

      @max_iters = nil
      @track_overall_min = false
      @overall_min_score = 1.0 / 0.0
      @overall_min_score_sample = nil

      @generate_samples = proc { raise "no generating function provided" }
      @score_sample     = proc {|sample| raise "no score function provided" }
      @estimate         = proc {|elite| raise "no estimate function provided" }
      @update           = proc {|estimated_params| estimated_params }
      @stop_decision    = proc {
        raise "no max_iters provided" unless self.max_iters
        self.num_iters >= self.max_iters
      }

      yield(self) if block_given?
    end

    attr_accessor :params

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

    #
    # Generic cross entropy routine.
    #
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

        # Compute new parameter estimates.
        estimated_params = @estimate.call(elite)

        # Update main parameter estimates.
        self.params = @update.call(estimated_params)

        @num_iters += 1
      end until @stop_decision.call
    end
  end
end

