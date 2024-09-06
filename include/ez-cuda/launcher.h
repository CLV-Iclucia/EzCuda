//
// Created by creeper on 9/2/24.
//

#ifndef GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_EZ_LAUNCHER_H
#define GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_EZ_LAUNCHER_H
#include <optional>
#include <cuda_runtime.h>
#include <ez-cuda/macros.h>
namespace ezcuda {

struct KernelLaunchingParams {
  size_t blockSize{256};
  size_t sharedMemBytes{0};
};

template<typename Derived>
struct ParallelLauncher {
  explicit ParallelLauncher(const KernelLaunchingParams &params = {}) : launcherConfig(params) {}
  Derived &wait() {
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
      throw std::runtime_error("CUDA error");
    }
    return static_cast<Derived &>(*this);
  }
protected:
  KernelLaunchingParams launcherConfig;
};

template<typename Func>
EZ_GLOBAL void kernelExec(int nThreads, Func func) {
  int i = tid(x);
  if (i >= nThreads)
    return;
  func(i);
}

struct ParallelFor : ParallelLauncher<ParallelFor> {
  explicit ParallelFor(const KernelLaunchingParams &params = {}) : ParallelLauncher(params) {}
  ParallelFor &setBlockSize(int specifiedBlockSize) {
    launcherConfig.blockSize = specifiedBlockSize;
    return *this;
  }
  ParallelFor &useAutoBlockSize() {
    launcherConfig.blockSize = 0;
    return *this;
  }
  ParallelFor &useSharedMemory(int bytes) {
    launcherConfig.sharedMemBytes = bytes;
    return *this;
  }

  template<typename Func>
  ParallelFor &apply(int nThreads, Func &&func) {
    size_t blockSize = launcherConfig.blockSize;
    if (!blockSize) {
      int minGridDim;
      int bestBlockSize;
      cudaOccupancyMaxPotentialBlockSize(&minGridDim,
                                         &bestBlockSize,
                                         &kernelExec<Func>,
                                         launcherConfig.sharedMemBytes,
                                         0);
      blockSize = static_cast<size_t>(bestBlockSize);
    }
    size_t numBlocks = (nThreads + blockSize - 1) / blockSize;
    kernelExec<<<numBlocks, blockSize, launcherConfig.sharedMemBytes>>>(nThreads, std::forward<Func>(func));
    return *this;
  }
};

template<typename T, typename Func>
EZ_GLOBAL void kernelArrayApply(int nThreads, CudaArrayViewer<T> viewer, Func func) {
  int i = tid(x);
  if (i >= nThreads)
    return;
  func(viewer[i]);
}

template<typename T>
struct ArrayForEach : ParallelLauncher<ArrayForEach<T>> {
  explicit ArrayForEach(CudaArray<T> &array) : array(array) {}
  template<typename Func>
  ArrayForEach &apply(Func &&func) {
    size_t nThreads = array.size();

  }

private:
  CudaArray<T> &array;
};
}
#endif //GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_EZ_LAUNCHER_H
