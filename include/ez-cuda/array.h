//
// Created by creeper on 9/2/24.
//

#ifndef GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_EZ_ARRAY_H
#define GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_EZ_ARRAY_H
#include <vector>
#include <span>
#include <cuda_runtime.h>
#include <ez-cuda/macros.h>
#include <ez-cuda/properties.h>
namespace ezcuda {

template<typename T>
struct CudaArrayViewer {
  T *ptr;
  size_t size;
  EZ_DEVICE EZ_FORCEINLINE const T &operator[](size_t idx) const {
    return ptr[idx];
  }
  EZ_DEVICE EZ_FORCEINLINE T &operator[](size_t idx) { return ptr[idx]; }
};

template<typename T>
struct ConstCudaArrayViewer {
  T *ptr;
  size_t size;
  EZ_DEVICE EZ_FORCEINLINE const T &operator[](size_t idx) const {
    return ptr[idx];
  }
};

template<typename T>
struct CudaArray {
  CudaArray() = default;
  DISABLE_COPY(CudaArray)
  CudaArray &operator=(CudaArray &&other) noexcept {
    if (this != &other) {
      m_size = other.m_size;
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }
  CudaArray(CudaArray &&other) noexcept
      : m_size(other.m_size), ptr(other.ptr) {
    other.ptr = nullptr;
  }

  EZ_CALLABLE explicit CudaArray(size_t size_) : m_size(size_) {
    cudaMalloc(&ptr, m_size * sizeof(T));
  }

  explicit CudaArray(const std::vector<T> &vec)
      : m_size(vec.size()), ptr(nullptr) {
    cudaMalloc(&ptr, m_size * sizeof(T));
    cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  explicit CudaArray(std::span<const T> span)
      : m_size(span.size()), ptr(nullptr) {
    cudaMalloc(&ptr, m_size * sizeof(T));
    cudaMemcpy(ptr, span.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  EZ_CALLABLE ~CudaArray() { cudaFree(ptr); }

  [[nodiscard]] T *data() const { return ptr; }

  [[nodiscard]] size_t size() const { return m_size; }

  CudaArray& copyFrom(std::span<const T> span) {
    if (m_size < span.size()) {
      cudaFree(ptr);
      cudaMalloc(&ptr, span.size() * sizeof(T));
    }
    cudaMemcpy(ptr, span.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
    return *this;
  }

  CudaArray& copyFrom(const std::vector<T> &vec) {
    if (m_size < vec.size()) {
      cudaFree(ptr);
      cudaMalloc(&ptr, vec.size() * sizeof(T));
    }
    cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
    return *this;
  }

  CudaArray& resize(size_t size_) {
    if (m_size == size_)
      return;
    if (ptr)
      cudaFree(ptr);
    m_size = size_;
    cudaMalloc(&ptr, m_size * sizeof(T));
    return *this;
  }

  void copyTo(std::vector<T> &vec) const {
    vec.resize(m_size);
    cudaMemcpy(vec.data(), ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
  }

  std::vector<T> toHost() const {
    std::vector<T> vec(m_size);
    cudaMemcpy(vec.data(), ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
    return vec;
  }

  EZ_HOST EZ_FORCEINLINE CudaArrayViewer<T> viewer() const {
    return {ptr};
  }

  EZ_HOST EZ_FORCEINLINE ConstCudaArrayViewer<T> cviewer() const {
    return {ptr};
  }

  EZ_DEVICE EZ_FORCEINLINE T &operator[](size_t idx) { return ptr[idx]; }

  EZ_DEVICE EZ_FORCEINLINE const T &operator[](size_t idx) const {
    return ptr[idx];
  }

  EZ_HOST EZ_FORCEINLINE T *begin() { return ptr; }

  EZ_HOST EZ_FORCEINLINE T *end() { return ptr + m_size; }

private:
  uint32_t m_size{};
  T *ptr{};
};

}
#endif
