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
              IT &NumThredsPBlock, IT &numStreams, bool& doCheck) {

  po::options_description desc(fmt::format("Usage : {} [OPTIONS]", argv[0]),
                               100);
  // clang-format off
    desc.add_options()
    ("help,h", "produce help message")
    ("num_elements,N", po::value<IT>(&N)->default_value(100), "set number of elements")
    ("reps,R", po::value<IT>(&NumReps)->default_value(16), "set num of kernel repetions")
    ("blocks,B", po::value<IT>(&NumBlocks)->default_value(CH::getSMCount() * 16), "set number of blocks")
    ("threads_per_block,T", po::value<IT>(&NumThredsPBlock)->default_value(CH::getWarpSize() * 16), "set number of threads per block")
    ("Streams,S", po::value<IT>(&numStreams)->default_value(4), "set number of threads per block")
    ("CHECK,C", po::bool_switch(&doCheck)->default_value(false), "set number of threads per block")
    ;
  // clang-format on
  po::variables_map vm;
  const auto parsedoptions = po::command_line_parser(argc, argv).options(desc).run();
  po::store(parsedoptions, vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc;
    exit(0);
  }
}

#endif