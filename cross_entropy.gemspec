# -*- encoding: utf-8 -*-
# frozen_string_literal: true

lib = File.expand_path('../lib/', __FILE__)
$LOAD_PATH.unshift lib unless $LOAD_PATH.include?(lib)

require 'cross_entropy/version'

Gem::Specification.new do |s|
  s.name              = 'cross_entropy'
  s.version           = CrossEntropy::VERSION
  s.platform          = Gem::Platform::RUBY
  s.authors           = ['John Lees-Miller']
  s.email             = ['jdleesmiller@gmail.com']
  s.homepage          = 'https://github.com/jdleesmiller/cross_entropy'
  s.summary = 'Solve optimisation problems with the Cross Entropy Method.'
  s.description = 'Includes solvers for continuous and discrete multivariate' \
    ' optimisation problems.'

  s.add_runtime_dependency 'narray', '~> 0.6'
  s.add_development_dependency 'gemma', '~> 4.1'

  s.files       = Dir.glob('{lib,bin}/**/*.rb') + %w(README.md)
  s.test_files  = Dir.glob('test/cross_entropy/*_test.rb')
  s.executables = Dir.glob('bin/*').map { |f| File.basename(f) }

  s.rdoc_options = [
    '--main',    'README.md',
    '--title',   "#{s.full_name} Documentation"
  ]
  s.extra_rdoc_files << 'README.md'
end
