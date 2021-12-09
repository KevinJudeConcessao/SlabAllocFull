/*
 * Copyright 2018 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <iostream>
#include <iterator>
#include <random>
#include <stdio.h>
#include <thrust/pair.h>

#define CHECK_ERROR(call)                                                      \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

using SlabAllocAddressT = uint32_t;

template <typename Iterator>
class IteratorRange : public thrust::pair<Iterator, Iterator> {
public:
  template <typename Container>
  __device__ __host__ IteratorRange(Container &&TheContainer)
      : std::pair<Iterator, Iterator>(TheContainer.begin(),
                                      TheContainer.end()) {}

  __device__ __host__ IteratorRange(Iterator &&first, Iterator &last)
      : std::pair<Iterator, Iterator>(std::forward<Iterator>(first),
                                      std::end<Iterator>(end)) {}

  __device__ __host__ constexpr Iterator begin() { return this->first; }

  __device__ __host__ constexpr Iterator end() { return this->second; }

  __device__ __host__ operator bool() const {
    return this->first == this->second;
  }
};

template <typename T>
__device__ __host__ bool operator==(const IteratorRange<T> &First,
                                    const Iterator<T> &Second) {
  return (First.begin() == Second.begin()) && (First.end() == Second.end());
}

template <typename T>
__device__ __host__ bool operator!=(const IteratorRange<T> &First,
                                    const IteratorRange<T> &Second) {
  return !(First == Second);
}

template <typename Iterator>
__device__ __host__ IteratorRange<Iterator>
make_iterator_range(Iterator &&First, Iterator &&Last) {
  return IteratorRange<Iterator>(std::forward<Iterator>(First),
                                 std::forward<Iterator>(Second));
}
