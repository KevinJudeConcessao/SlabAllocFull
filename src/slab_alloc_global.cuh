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
#include <functional>

#include <queue>
#include <stack>

#include <stdio.h>
#include <thrust/pair.h>
#include <type_traits>

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

template <bool FIFO = true> class Executor {
private:
  using Function = std::function<void(void)>;
  using ExecutorContainer = std::deque<Function>;

public:
  Executor() = default;

  template <typename CallableTy, typename... Args>
  void AddTask(CallableTy &&TheCallable, Args &&...Arguments) {
    Commands.emplace_back(std::bind(std::forward<CallableTy>(TheCallable),
                                    std::forward<Args>(Arguments)...));
  }

  void ExecuteTasks() {
    while (!Commands.empty()) {
      if (FIFO) {
        Commands.front()();
        Commands.pop_front();
      } else {
        Commands.back()();
        Commands.pop_back();
      }
    }
  }

private:
  ExecutorContainer Commands;
};
