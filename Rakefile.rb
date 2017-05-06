# frozen_string_literal: true
require 'rubygems'
require 'bundler/setup'
require 'gemma'

Gemma::RakeTasks.with_gemspec_file 'cross_entropy.gemspec'

task default: :test
