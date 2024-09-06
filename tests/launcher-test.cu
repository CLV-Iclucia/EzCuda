//
// Created by creeper on 9/5/24.
//
#include <iostream>
#include <ez-cuda/array.h>
#include <ez-cuda/launcher.h>
using namespace ezcuda;
int main() {
  CudaArray<int> a({1, 2, 3, 4, 5});
  CudaArray<int> b({1, 2, 3, 4, 5});
  CudaArray<int> c({0, 0, 0, 0, 0});
  ParallelFor().useAutoBlockSize()
      .apply(5,
             [
                 a = a.cviewer(),
                 b = b.cviewer(),
                 c = c.viewer()
             ] EZ_DEVICE(int i) mutable {
    c[i] = a[i] + b[i];
  }).wait();
  std::vector<int> result;
  c.copyTo(result);
  for (int i = 0; i < 5; i++)
    std::cout << "c[" << i << "] = " << result[i] << "\n";
  for (int i = 0; i < 5; i++) {
    if (result[i] != 2 * (i + 1)) {
      std::cerr << "failed at " << i << " expect " << 2 * (i + 1) << " but got " << result[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Test passed" << std::endl;
  return 0;
}