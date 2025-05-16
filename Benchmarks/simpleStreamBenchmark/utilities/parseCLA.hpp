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

using IT = size_t;

void parseCLA(int argc, char const *argv[], IT &N, IT &NumReps, IT &NumBlocks,
              IT &NumThredsPBlock, bool &verbose,
              bool &doCheck) {
  try {

    po::options_description desc(fmt::format("Usage : {} [OPTIONS]", argv[0]),
                                 100);
    // clang-format off
  desc.add_options()
  ("help,h", "produce help message")
  ("num_elements,N", po::value<IT>(&N)->default_value(1<<25), "set number of elements")
  ("reps,R", po::value<IT>(&NumReps)->default_value(16), "set num of kernel repetions")
  ("blocks,B", po::value<IT>(&NumBlocks)->default_value(CH::getSMCount()), "set number of blocks")
  ("threads_per_block,T", po::value<IT>(&NumThredsPBlock)->default_value(CH::getWarpSize()), "set number of threads per block")
  ("VERBOSE,V", po::bool_switch(&verbose)->default_value(false), "print full performance timings/bandwidths")
  ("CHECK,C", po::bool_switch(&doCheck)->default_value(true), "perform check to verify kernel correctness")
  ;
    // clang-format on
    po::variables_map vm;
    const auto parsedoptions =
      po::command_line_parser(argc, argv).options(desc).run();
    po::store(parsedoptions, vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc;
      exit(EXIT_SUCCESS);
    }
  } catch (std::exception &e) {
    fmt::print("Error : {}\n", e.what());
    exit(1);
  }
}

#endif