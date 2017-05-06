# frozen_string_literal: true

require 'rubygems'
require 'bundler/setup'
require 'gemma'
require 'rubocop/rake_task'

Gemma::RakeTasks.with_gemspec_file 'cross_entropy.gemspec'
RuboCop::RakeTask.new(:lint)

task default: :test

task ci: %i[test lint]
