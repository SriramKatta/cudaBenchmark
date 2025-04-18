#ifndef PARSECLA_HPP
#define PARSECLA_HPP
#pragma once

#include <fmt/format.h>
#include <boost/program_options.hpp>
#include <iostream>
namespace po = boost::program_options;
void parseCLA(int argc, char const *argv[], size_t &N, size_t &NumReps,
              size_t &NumBlocks, size_t &NumThredsPBlock) {
  po::options_description desc("Allowed Options");
  // clang-format off
    desc.add_options()
    ("help,h", "produce help message")
    ("num_elements,N", po::value<size_t>(&N)->default_value(100), "set number of elements")
    ("reps,R", po::value<size_t>(&NumReps)->default_value(16), "set num of kernel repetions")
    ("blocks,B", po::value<size_t>(&NumBlocks)->default_value(30), "set number of blocks")
    ("threads_per_block,T", po::value<size_t>(&NumThredsPBlock)->default_value(128), "set number of threads per block")
    ;
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("num_elements")) {
    N = vm["num_elements"].as<size_t>();
  }
  if (vm.count("reps")) {
    NumReps = vm["reps"].as<size_t>();
  }
  if (vm.count("blocks")) {
    NumBlocks = vm["blocks"].as<size_t>();
  }
  if (vm.count("threads_per_block")) {
    NumThredsPBlock = vm["threads_per_block"].as<size_t>();
  }
  if (vm.count("help") || argc == 1) {
    std::cout << desc;
    fmt::print(
      "N : {} | Nreps : {} | NumBlocks : {} | NumthreadsPblocks : {}\n", N,
      NumReps, NumBlocks, NumThredsPBlock);
  }
}

#endif