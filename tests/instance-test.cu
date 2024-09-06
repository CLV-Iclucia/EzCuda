//
// Created by creeper on 9/5/24.
//
#include <iostream>
#include <ez-cuda/instance.h>
#include <ez-cuda/array.h>
#include <ez-cuda/launcher.h>
using namespace ezcuda;

int main() {
  CudaInstance<int> a(10);
  CudaArray<int> b(32);
  ParallelFor().useAutoBlockSize().apply(5,[
        a = a.cviewer(),
        b = b.viewer()
      ] EZ_DEVICE (int i) mutable {
        b[i] = *a;
  }).wait();
  auto host = b.toHost();
  for (int i = 0; i < 5; i++) {
    std::cout << host[i] << std::endl;
  }
}