//
// Created by creeper on 9/5/24.
//

#ifndef EZCUDA_INCLUDE_EZ_CUDA_PROPERTIES_H
#define EZCUDA_INCLUDE_EZ_CUDA_PROPERTIES_H

#define DISABLE_COPY(Class) \
  Class(const Class &) = delete; \
  Class &operator=(const Class &) = delete;

#define RESOURCE(Class) \
  Class(Class&& ) = delete;

#endif //EZCUDA_INCLUDE_EZ_CUDA_PROPERTIES_H
