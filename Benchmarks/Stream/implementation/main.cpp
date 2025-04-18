#include "parseCLA.hpp"

int main(int argc, char const *argv[]) 
{
  size_t N;
  size_t NumReps;
  size_t NumBlocks;
  size_t NumThredsPBlock;

  parseCLA(argc, argv, N, NumReps, NumBlocks, NumThredsPBlock);


  return 0;
}
