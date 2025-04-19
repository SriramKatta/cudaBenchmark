#ifndef PARSECLA_HPP
#define PARSECLA_HPP
#pragma once

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include <iostream>

#include "cuda_helper.cuh"


namespace po = boost::program_options;
namespace CH = cuda_helpers;

template <typename IT>
void parseCLA(int argc, char const *argv[], IT &N, IT &NumReps, IT &NumBlocks,
              IT &NumThredsPBlock, IT &numStreams) {


  po::options_description desc("Allowed Options", 100);
  // clang-format off
    desc.add_options()
    ("help,h", "produce help message")
    ("num_elements,N", po::value<IT>(&N)->default_value(100), "set number of elements")
    ("reps,R", po::value<IT>(&NumReps)->default_value(16), "set num of kernel repetions")
    ("blocks,B", po::value<IT>(&NumBlocks)->default_value(CH::getSMCount() * 16), "set number of blocks")
    ("threads_per_block,T", po::value<IT>(&NumThredsPBlock)->default_value(CH::getWarpSize() * 16), "set number of threads per block")
    ("Streams,S", po::value<IT>(&numStreams)->default_value(4), "set number of threads per block")
    ;
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  if (vm.count("num_elements")) {
    N = vm["num_elements"].as<IT>();
  }
  if (vm.count("reps")) {
    NumReps = vm["reps"].as<IT>();
  }
  if (vm.count("blocks")) {
    NumBlocks = vm["blocks"].as<IT>();
  }
  if (vm.count("threads_per_block")) {
    NumThredsPBlock = vm["threads_per_block"].as<IT>();
  }
  if (vm.count("Streams")) {
    numStreams = vm["Streams"].as<IT>();
  }
  if (vm.count("help")) {
    std::cout << desc;
    exit(0);
  }
}

#endif