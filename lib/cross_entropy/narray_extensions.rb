# frozen_string_literal: true
module CrossEntropy
  #
  # Some extensions to NArray.
  #
  # Note that I've opened a pull request for general cumsum and tile, but it's
  # still open without comment after three years, and I think they have stopped
  # working on this version of narray.
  # https://github.com/masa16/narray/pull/7
  #
  module NArrayExtensions
    refine NArray do
      #
      # Cumulative sum along dimension +dim+; modifies this array in place.
      #
      # @param [Number] dim non-negative
      #
      # @return [NArray] self
      #
      def cumsum_general!(dim = 0)
        if self.dim > dim
          if self.dim == 1
            # use the built-in version for dimension 1
            cumsum_1!
          else
            # for example, if this is a matrix and dim = 0, mask_0 selects the
            # first column of the matrix and mask_1 selects the second column;
            # then we just shuffle them along and accumulate.
            mask_0 = (0...self.dim).map { |d| d == dim ? 0 : true }
            mask_1 = (0...self.dim).map { |d| d == dim ? 1 : true }
            while mask_1[dim] < shape[dim]
              self[*mask_1] += self[*mask_0]
              mask_0[dim] += 1
              mask_1[dim] += 1
            end
          end
        end
        self
      end

      #
      # Cumulative sum along dimension +dim+.
      #
      # @param [Number] dim non-negative
      #
      # @return [NArray]
      #
      def cumsum_general(dim = 0)
        dup.cumsum_general!(dim)
      end

      # The built-in cumsum only does vectors (dim 1).
      alias_method :cumsum_1, :cumsum
      alias_method :cumsum, :cumsum_general
      alias_method :cumsum_1!, :cumsum!
      alias_method :cumsum!, :cumsum_general!

      #
      # Replicate this array to make a tiled array; this is the matlab function
      # repmat.
      #
      # @param [Array<Number>] reps number of times to repeat in each dimension;
      # note that reps.size is allowed to be different from self.dim, and
      # dimensions of size 1 will be added to compensate
      #
      # @return [NArray] with same typecode as self
      #
      def tile(*reps)
        if dim == 0 || reps.member?(0)
          # Degenerate case: 0 dimensions or dimension 0
          res = NArray.new(typecode, 0)
        else
          if reps.size <= dim
            # Repeat any extra dims once.
            reps += [1] * (dim - reps.size)
            tile = self
          else
            # Have to add some more dimensions (with implicit shape[dim] = 1).
            tile_shape = shape + [1] * (reps.size - dim)
            tile = reshape(*tile_shape)
          end

          # Allocate tiled matrix.
          res_shape = (0...tile.dim).map { |i| tile.shape[i] * reps[i] }
          res = NArray.new(typecode, *res_shape)

          # Copy tiles.
          # This probably isn't the most efficient way of doing this; just doing
          # res[] = tile doesn't seem to work in general
          nested_for_zero_to(reps) do |tile_pos|
            tile_slice = (0...tile.dim).map do |i|
              start_index = tile.shape[i] * tile_pos[i]
              end_index = tile.shape[i] * (tile_pos[i] + 1)
              start_index...end_index
            end
            res[*tile_slice] = tile
          end
        end
        res
      end

      #
      # Convert a linear (1D) index into subscripts for an array with the given
      # shape; this is the matlab function ind2sub.
      #
      # (TODO: There must be a function in NArray to do this, but I can't find
      # it.)
      #
      # @param [Integer] index non-negative
      #
      # @return [Array<Integer>] subscript corresponding to the given linear
      #         index; this is the same size as +shape+
      #
      def index_to_subscript(index)
        if index >= size
          raise \
            IndexError,
            "out of bounds: index=#{index} for shape=#{shape.inspect}"
        end

        shape.map do |s|
          index, r = index.divmod(s)
          r
        end
      end

      #
      # Sample from an array that represents an empirical probability mass
      # function (pmf). It is assumed that this is an array of probabilities,
      # and that the sum over the whole array is one (up to rounding error). An
      # index into the array is chosen in proportion to its probability.
      #
      # @example select a subscript uniform-randomly
      #   NArray.float(3,3,3).fill!(1).div!(3*3*3).sample_pmf #=> [2, 2, 0]
      #
      # @param [NArray] r if you have already generated the random sample, you
      #        can pass it in here; if nil, a random sample will be generated;
      #        this is used for testing; must be have shape <tt>[1]</tt> if
      #        specified
      #
      # @return [Array<Integer>] subscripts of a randomly selected into the
      #         array; this is the same size as +shape+
      #
      def sample_pmf(r = nil)
        index_to_subscript(flatten.sample_pmf_dim(0, r))
      end

      #
      # Sample from an array in which the given dimension, +dim+, represents an
      # empirical probability mass function (pmf). It is assumed that the
      # entries along +dim+ are probabilities that sum to one (up to rounding
      # error).
      #
      # @example a matrix in which dim 0 sums to 1
      #   NArray[[0.1,0.2,0.7],
      #          [0.3,0.5,0.2],
      #          [0.0,0.2,0.8],
      #          [0.7,0.1,0.2]].sample_pmf(1)
      #   #=> NArray.int(2) [ 1, 1, 2, 0 ] # random indices into dimension 1
      #
      # @param [Integer] dim dimension to sample along
      #
      # @param [NArray] r if you have already generated the random sample, you
      #        can pass it in here; if nil, a random sample will be generated;
      #        this is used for testing; see also sample_cdf_dim
      #
      # @return [NArray] integer subscripts
      #
      def sample_pmf_dim(dim = 0, r = nil)
        cumsum(dim).sample_cdf_dim(dim, r)
      end

      #
      # Sample from an array in which the given dimension, +dim+, represents an
      # empirical cumulative distribution function (cdf). It is assumed that the
      # entries along +dim+ are sums of probabilities, and that the last entry
      # along dim should be 1 (up to rounding error)
      #
      # @param [Integer] dim dimension to sample along
      #
      # @param [NArray] r if you have already generated the random sample, you
      #        can pass it in here; if nil, a random sample will be generated;
      #        this is used for testing; see also sample_cdf_dim
      #
      # @return [NArray] integer subscripts
      #
      def sample_cdf_dim(dim = 0, r = nil)
        raise 'self.dim must be > dim' unless self.dim > dim

        # generate random sample, unless one was given for testing
        r_shape = (0...self.dim).map { |i| i == dim ? 1 : shape[i] }
        r = NArray.new(typecode, *r_shape).random! unless r

        # allocate space for results -- same size as the random sample
        res = NArray.int(*r_shape)

        # for every other dimension, look for the first element that is over the
        # threshold
        nested_for_zero_to(r_shape) do |slice|
          r_thresh    = r[*slice]
          res[*slice] = shape[dim] - 1 # default to last
          self_slice = slice.dup
          for self_slice[dim] in 0...shape[dim]
            if r_thresh < self[*self_slice]
              res[*slice] = self_slice[dim]
              break
            end
          end
        end

        res[*(0...self.dim).map { |i| i == dim ? 0 : true }]
      end

      private

      #
      # This is effectively <tt>suprema.size</tt> nested 'for' loops, in which
      # the outermost loop runs over <tt>0...suprema.first</tt>, and the
      # innermost loop runs over <tt>0...suprema.last</tt>.
      #
      # For example, when +suprema+ is [3], it yields [0], [1] and [2], and when
      # +suprema+ is [3,2] it yields [0,0], [0,1], [1,0], [1,1], [2,0] and
      # [2,1].
      #
      # @param [Array<Integer>] suprema non-negative entries; does not yield if
      #        empty
      #
      # @return [nil]
      #
      def nested_for_zero_to(suprema)
        unless suprema.empty?
          nums = suprema.map { |n| (0...n).to_a }
          nums.first.product(*nums.drop(1)).each do |num|
            yield num
          end
        end
        nil
      end
    end
  end
end
