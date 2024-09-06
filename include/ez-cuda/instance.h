//
// Created by creeper on 9/5/24.
//

#ifndef EZCUDA_INCLUDE_EZ_CUDA_INSTANCE_H
#define EZCUDA_INCLUDE_EZ_CUDA_INSTANCE_H
#include <cuda_runtime.h>
#include <ez-cuda/properties.h>
namespace ezcuda {
template<typename T>
struct InstanceViewer {
  explicit InstanceViewer(T *ptr) : ref(*ptr) {}
  T *operator->() {
    return &ref;
  }
  const T *operator->() const {
    return &ref;
  }
  T &operator*() {
    return ref;
  }
  const T &operator*() const {
    return ref;
  }
private:
  T &ref;
};

template<typename T>
struct ConstInstanceViewer {
  explicit ConstInstanceViewer(const T *ptr) : ref(*ptr) {}
  const T *operator->() const {
    return &ref;
  }
  const T &operator*() const {
    return ref;
  }
private:
  const T &ref;
};

template<typename T>
struct CudaInstance {
  DISABLE_COPY(CudaInstance)
  explicit CudaInstance(const T &value) {
    cudaMalloc(&ptr, sizeof(T));
    cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
  }
  InstanceViewer<T> viewer() {
    return InstanceViewer<T>(ptr);
  }
  ConstInstanceViewer<T> cviewer() const {
    return ConstInstanceViewer<T>{ptr};
  }
  CudaInstance &assign(const T &value) {
    if (ptr == nullptr)
      cudaMalloc(&ptr, sizeof(T));
    cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
    return *this;
  }
  CudaInstance &passTo(T &value) {
    cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return *this;
  }
  T toHost() const {
    T value;
    cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
  }
  ~CudaInstance() {
    cudaFree(ptr);
  }
private:
  T *ptr{};
};
}
#endif //EZCUDA_INCLUDE_EZ_CUDA_INSTANCE_H
