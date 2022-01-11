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
#include "slab_alloc_global.cuh"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>

#define MAX_SUPERBLOCK_ALLOCATIONS 4096

/*
 * This class does not own any memory, and will be shallowly copied into device
 * kernel
 */
template <uint32_t LOG_NUM_MEM_BLOCKS_, uint32_t NUM_SUPER_BLOCKS_ALLOCATOR_,
          uint32_t MEM_UNIT_WARP_MULTIPLES_ = 1>
class SlabAllocLightContext {
public:
  // fixed parameters for the SlabAllocLight
  static constexpr uint32_t NUM_MEM_UNITS_PER_BLOCK_ = 1024;
  static constexpr uint32_t NUM_BITMAP_PER_MEM_BLOCK_ = 32;
  static constexpr uint32_t BITMAP_SIZE_ = 32;
  static constexpr uint32_t WARP_SIZE_ = 32;
  static constexpr uint32_t MEM_UNIT_SIZE_ =
      MEM_UNIT_WARP_MULTIPLES_ * WARP_SIZE_;
  static constexpr uint32_t SUPER_BLOCK_BIT_OFFSET_ALLOC_ = 27;
  static constexpr uint32_t MEM_BLOCK_BIT_OFFSET_ALLOC_ = 10;
  static constexpr uint32_t MEM_UNIT_BIT_OFFSET_ALLOC_ = 5;
  static constexpr uint32_t NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ =
      (1 << LOG_NUM_MEM_BLOCKS_);
  static constexpr uint32_t MEM_BLOCK_SIZE_ =
      NUM_MEM_UNITS_PER_BLOCK_ * MEM_UNIT_SIZE_;
  static constexpr uint32_t SUPER_BLOCK_SIZE_ =
      ((BITMAP_SIZE_ + MEM_BLOCK_SIZE_) * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
  static constexpr uint32_t MEM_BLOCK_OFFSET_ =
      (BITMAP_SIZE_ * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
  static constexpr uint32_t num_super_blocks_ = NUM_SUPER_BLOCKS_ALLOCATOR_;

  __device__ __host__ SlabAllocLightContext()
      : d_super_blocks_(nullptr), hash_coef_(0), num_attempts_(0),
        resident_index_(0), super_block_index_(0), allocated_index_(0) {}

  __device__ __host__ SlabAllocLightContext(const SlabAllocLightContext &rhs)
      : d_super_blocks_(rhs.d_super_blocks_), hash_coef_(rhs.hash_coef_),
        num_attempts_(0), resident_index_(0), super_block_index_(0),
        allocated_index_(0) {}

  SlabAllocLightContext &operator=(const SlabAllocLightContext &rhs) {
    d_super_blocks_ = rhs.d_super_blocks_;
    hash_coef_ = rhs.hash_coef_;
    num_attempts_ = 0;
    resident_index_ = 0;
    super_block_index_ = 0;
    allocated_index_ = 0;
    return *this;
  }

  __device__ __host__ ~SlabAllocLightContext(){};

  __device__ __host__ void initParameters(uint32_t *d_super_block,
                                          uint32_t hash_coef) {
    d_super_blocks_ = d_super_block;
    hash_coef_ = hash_coef;
  }

  __device__ __host__ void copyParameters(const SlabAllocLightContext &rhs) {
    d_super_blocks_ = rhs.d_super_blocks_;
    hash_coef_ = rhs.hash_coef_;
  }

  // =========
  // some helper inline address functions:
  // =========
  __device__ __host__ __forceinline__ uint32_t
  getSuperBlockIndex(SlabAllocAddressT address) const {
    return address >> SUPER_BLOCK_BIT_OFFSET_ALLOC_;
  }

  __device__ __host__ __forceinline__ uint32_t
  getMemBlockIndex(SlabAllocAddressT address) const {
    return ((address >> MEM_BLOCK_BIT_OFFSET_ALLOC_) & 0x1FFFF);
  }

  __device__ __host__ __forceinline__ SlabAllocAddressT
  getMemBlockAddress(SlabAllocAddressT address) const {
    return (MEM_BLOCK_OFFSET_ + getMemBlockIndex(address) * MEM_BLOCK_SIZE_);
  }

  __device__ __host__ __forceinline__ uint32_t
  getMemUnitIndex(SlabAllocAddressT address) const {
    return address & 0x3FF;
  }

  __device__ __host__ __forceinline__ SlabAllocAddressT
  getMemUnitAddress(SlabAllocAddressT address) {
    return getMemUnitIndex(address) * MEM_UNIT_SIZE_;
  }

  __device__ __forceinline__ uint32_t *
  getPointerFromSlab(const SlabAllocAddressT &next, const uint32_t &laneId) {
    return reinterpret_cast<uint32_t *>(d_super_blocks_) +
           addressDecoder(next) + laneId;
  }

  __device__ __forceinline__ uint32_t *
  getPointerForBitmap(const uint32_t super_block_index,
                      const uint32_t bitmap_index) {
    return d_super_blocks_ + super_block_index * SUPER_BLOCK_SIZE_ +
           bitmap_index;
  }

  // called at the beginning of the kernel:
  __device__ __forceinline__ void createMemBlockIndex(uint32_t global_warp_id) {
    super_block_index_ = global_warp_id % num_super_blocks_;
    resident_index_ =
        (hash_coef_ * global_warp_id) >> (32 - LOG_NUM_MEM_BLOCKS_);
  }

  // called when the allocator fails to find an empty unit to allocate:
  __device__ __forceinline__ void updateMemBlockIndex(uint32_t global_warp_id) {
    num_attempts_++;
    super_block_index_++;
    super_block_index_ =
        (super_block_index_ == num_super_blocks_) ? 0 : super_block_index_;
    resident_index_ = (hash_coef_ * (global_warp_id + num_attempts_)) >>
                      (32 - LOG_NUM_MEM_BLOCKS_);
    // loading the assigned memory block:
    resident_bitmap_ =
        *((d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_) +
          resident_index_ * BITMAP_SIZE_ + (threadIdx.x & 0x1f));
  }

  // Objective: each warp selects its own resident warp allocator:
  __device__ __forceinline__ void initAllocator(uint32_t &tid,
                                                uint32_t &laneId) {
    // hashing the memory block to be used:
    createMemBlockIndex(tid >> 5);

    // loading the assigned memory block:
    resident_bitmap_ =
        *(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
          resident_index_ * BITMAP_SIZE_ + laneId);
    allocated_index_ = 0xFFFFFFFF;
  }

  __device__ __forceinline__ uint32_t warpAllocate(const uint32_t &laneId) {
    // tries and allocate a new memory units within the resident memory block
    // if it returns 0xFFFFFFFF, then there was not any empty memory unit
    // a new resident block should be chosen, and repeat again
    // allocated result:  5  bits: super_block_index
    //                    17 bits: memory block index
    //                    5  bits: memory unit index (hi-bits of 10bit)
    //                    5  bits: memory unit index (lo-bits of 10bit)
    int empty_lane = -1;
    uint32_t free_lane;
    uint32_t read_bitmap = resident_bitmap_;
    uint32_t allocated_result = 0xFFFFFFFF;
    // works as long as <31 bit are used in the allocated_result
    // in other words, if there are 32 super blocks and at most 64k blocks per
    // super block

    while (allocated_result == 0xFFFFFFFF) {
      empty_lane = __ffs(~resident_bitmap_) - 1;
      free_lane = __ballot_sync(0xFFFFFFFF, empty_lane >= 0);
      if (free_lane == 0) {
        // all bitmaps are full: need to be rehashed again:
        updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >> 5);
        read_bitmap = resident_bitmap_;
        continue;
      }
      uint32_t src_lane = __ffs(free_lane) - 1;
      if (src_lane == laneId) {
        read_bitmap =
            atomicCAS(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                          resident_index_ * BITMAP_SIZE_ + laneId,
                      resident_bitmap_, resident_bitmap_ | (1 << empty_lane));
        if (read_bitmap == resident_bitmap_) {
          // successful attempt:
          resident_bitmap_ |= (1 << empty_lane);
          allocated_result =
              (super_block_index_ << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
              (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
              (laneId << MEM_UNIT_BIT_OFFSET_ALLOC_) | empty_lane;
        } else {
          // Not successful: updating the current bitmap
          resident_bitmap_ = read_bitmap;
        }
      }
      // asking for the allocated result;
      allocated_result = __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
    }
    return allocated_result;
  }

  __device__ __forceinline__ uint32_t warpAllocateBulk(uint32_t &laneId,
                                                       const uint32_t k) {
    // tries and allocate k consecutive memory units within the resident memory
    // block if it returns 0xFFFFFFFF, then there was not any empty memory unit
    // a new resident block should be chosen, and repeat again
    // allocated result:  5  bits: super_block_index
    //                    17 bits: memory block index
    //                    5  bits: memory unit index (hi-bits of 10bit)
    //                    5  bits: memory unit index (lo-bits of 10bit)
    int empty_lane = -1;
    uint32_t free_lane;
    uint32_t read_bitmap = resident_bitmap_;
    uint32_t allocated_result = 0xFFFFFFFF;
    // works as long as <31 bit are used in the allocated_result
    // in other words, if there are 32 super blocks and at most 64k blocks per
    // super block

    while (allocated_result == 0xFFFFFFFF) {
      empty_lane =
          32 - (__ffs(__brev(
                   ~resident_bitmap_))); // reversing the order of assigning
                                         // lanes compared to single allocations
      const uint32_t mask = ((1 << k) - 1) << (empty_lane - k + 1);
      // mask = %x\n", context.resident_bitmap, empty_lane, mask);
      free_lane = __ballot_sync(
          0xFFFFFFFF,
          (empty_lane >= (k - 1)) &&
              !(resident_bitmap_ &
                mask)); // update true statement to make sure everything fits
      if (free_lane == 0) {
        // all bitmaps are full: need to be rehashed again:
        updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >> 5);
        read_bitmap = resident_bitmap_;
        continue;
      }
      uint32_t src_lane = __ffs(free_lane) - 1;

      if (src_lane == laneId) {
        read_bitmap =
            atomicCAS(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                          resident_index_ * BITMAP_SIZE_ + laneId,
                      resident_bitmap_, resident_bitmap_ | mask);
        if (read_bitmap == resident_bitmap_) {
          // successful attempt:
          resident_bitmap_ |= mask;
          allocated_result =
              (super_block_index_ << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
              (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
              (laneId << MEM_UNIT_BIT_OFFSET_ALLOC_) | empty_lane;
        } else {
          // Not successful: updating the current bitmap
          resident_bitmap_ = read_bitmap;
        }
      }
      // asking for the allocated result;
      allocated_result = __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
    }
    return allocated_result;
  }

  /*
  This function, frees a recently allocated memory unit by a single thread.
  Since it is untouched, there shouldn't be any worries for the actual memory
  contents to be reset again.
*/
  __device__ __forceinline__ void freeUntouched(SlabAllocAddressT ptr) {
    atomicAnd(d_super_blocks_ + getSuperBlockIndex(ptr) * SUPER_BLOCK_SIZE_ +
                  getMemBlockIndex(ptr) * BITMAP_SIZE_ +
                  (getMemUnitIndex(ptr) >> 5),
              ~(1 << (getMemUnitIndex(ptr) & 0x1F)));
  }

  __host__ __device__ __forceinline__ SlabAllocAddressT
  addressDecoder(SlabAllocAddressT address_ptr_index) {
    return getSuperBlockIndex(address_ptr_index) * SUPER_BLOCK_SIZE_ +
           getMemBlockAddress(address_ptr_index) +
           getMemUnitIndex(address_ptr_index) * MEM_UNIT_WARP_MULTIPLES_ *
               WARP_SIZE_;
  }

  __host__ __device__ __forceinline__ void
  print_address(SlabAllocAddressT address_ptr_index) {
    printf("Super block Index: %d, Memory block index: %d, Memory unit index: "
           "%d\n",
           getSuperBlockIndex(address_ptr_index),
           getMemBlockIndex(address_ptr_index),
           getMemUnitIndex(address_ptr_index));
  }

  __device__ __forceinline__ void print_info() {
    printf(
        "SlabAllocLightContext: Thread: %d, d_super_blocks_: %p, hash_coef_: "
        "%u, "
        "num_attempts_: %u, "
        "resident_index_: %u, resident_bitmap_: %x, super_block_index_: %u\n",
        threadIdx.x, d_super_blocks_, hash_coef_, num_attempts_,
        resident_index_, resident_bitmap_, super_block_index_,
        allocated_index_);
  }

private:
  // a pointer to each super-block
  uint32_t *d_super_blocks_;

  // hash_coef (register): used as (16 bits, 16 bits) for hashing
  uint32_t hash_coef_; // a random 32-bit

  // resident_index: (register)
  // should indicate what memory block and super block is currently resident
  // (16 bits       + 5 bits)
  // (memory block  + super block)
  uint32_t num_attempts_;
  uint32_t resident_index_;
  uint32_t resident_bitmap_;
  uint32_t super_block_index_;
  uint32_t allocated_index_; // to be asked via shuffle after
};

/*
 * This class owns the memory for the allocator on the device
 */
template <uint32_t LOG_NUM_MEM_BLOCKS_, uint32_t NUM_SUPER_BLOCKS_ALLOCATOR_,
          uint32_t MEM_UNIT_WARP_MULTIPLES_ = 1>
class SlabAllocLight {
private:
  // a pointer to each super-block
  uint32_t *d_super_blocks_;

  // hash_coef (register): used as (16 bits, 16 bits) for hashing
  uint32_t hash_coef_; // a random 32-bit

  // the context class is actually copied shallowly into GPU device
  SlabAllocLightContext<LOG_NUM_MEM_BLOCKS_, NUM_SUPER_BLOCKS_ALLOCATOR_,
                        MEM_UNIT_WARP_MULTIPLES_>
      slab_alloc_context_;

public:
  // =========
  // constructor:
  // =========
  SlabAllocLight() : d_super_blocks_(nullptr), hash_coef_(0) {
    // random coefficients for allocator's hash function
    std::mt19937 rng(time(0));
    hash_coef_ = rng();

    // In the light version, we put num_super_blocks super blocks within a
    // single array
    CHECK_ERROR(cudaMalloc((void **)&d_super_blocks_,
                           slab_alloc_context_.SUPER_BLOCK_SIZE_ *
                               slab_alloc_context_.num_super_blocks_ *
                               sizeof(uint32_t)));

    for (int i = 0; i < slab_alloc_context_.num_super_blocks_; i++) {
      // setting bitmaps into zeros:
      CHECK_ERROR(cudaMemset(
          d_super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_, 0x00,
          slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
              slab_alloc_context_.BITMAP_SIZE_ * sizeof(uint32_t)));
      // setting empty memory units into ones:
      CHECK_ERROR(cudaMemset(
          d_super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_ +
              (slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
               slab_alloc_context_.BITMAP_SIZE_),
          0xFF,
          slab_alloc_context_.MEM_BLOCK_SIZE_ *
              slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
              sizeof(uint32_t)));
    }

    // initializing the slab context:
    slab_alloc_context_.initParameters(d_super_blocks_, hash_coef_);
  }

  // =========
  // destructor:
  // =========
  ~SlabAllocLight() { CHECK_ERROR(cudaFree(d_super_blocks_)); }

  // =========
  // Helper member functions:
  // =========
  SlabAllocLightContext<LOG_NUM_MEM_BLOCKS_, NUM_SUPER_BLOCKS_ALLOCATOR_,
                        MEM_UNIT_WARP_MULTIPLES_> *
  getContextPtr() {
    return &slab_alloc_context_;
  }
};

__constant__ uint32_t *SuperBlocks[MAX_SUPERBLOCK_ALLOCATIONS];
static uint32_t NextAllocationOffset = 0;

template <uint32_t, uint32_t, uint32_t> class SlabAlloc;

template <uint32_t MemoryBlocksLogN, uint32_t SuperBlocksN,
          uint32_t WordsPerMemUnit = 1>
class SlabAllocContext {
public:
  static constexpr uint32_t MemoryUnitsPerMemoryBlock = 1024;
  static constexpr uint32_t WarpSize = 32;
  static constexpr uint32_t BitMapSize = 32;
  static constexpr uint32_t BitMapsPerMemoryBlock = 32;
  static constexpr uint32_t NumberOfSuperBlocks = SuperBlocksN;
  static constexpr uint32_t WordsPerMemoryUnit = WordsPerMemUnit;
  static constexpr uint32_t MemoryBlocksPerSuperBlock = (1 << MemoryBlocksLogN);

  using MemoryUnit = uint32_t[WarpSize][WordsPerMemoryUnit];
  using MemoryBlock = MemoryUnit[MemoryUnitsPerMemoryBlock];
  using MemoryBlocks = MemoryBlock[MemoryBlocksPerSuperBlock];

  using MemoryBlockBitMap = uint32_t[WarpSize];
  using BitMap = MemoryBlockBitMap[MemoryBlocksPerSuperBlock];

  struct SuperBlock {
    BitMap TheBitMap;
    MemoryBlocks TheMemoryBlocks;
  };

  __device__ __host__ SlabAllocContext()
      : HashCoefficient{0}, NumberOfAttempts{0}, ResidentIndex{0},
        ResidentBitMap{0}, SuperBlockIndex{0}, SBAllocOffset{0} {}

  __device__ __host__ SlabAllocContext(const SlabAllocContext &SAC)
      : HashCoefficient(SAC.HashCoefficient), NumberOfAttempts{0},
        ResidentIndex{0}, ResidentBitMap{0}, SuperBlockIndex{0},
        SBAllocOffset{SAC.SBAllocOffset} {}

  SlabAllocContext &operator=(const SlabAllocContext &SAC) {
    HashCoefficient = SAC.HashCoefficient;
    NumberOfAttempts = 0;
    ResidentIndex = 0;
    SuperBlockIndex = 0;
    SBAllocOffset = SAC.SBAllocOffset;
    return *this;
  }

  __device__ __host__ ~SlabAllocContext() {}

private:
  /* Some Helper Functions */

  /* Structure of SlabAllocAddressT:
   *
   * │ 31           24 │ 23            10 │ 9             0 │
   * ┌─────────────────┬──────────────────┬─────────────────┐
   * │ SuperBlockIndex │ MemoryBlockIndex │ MemoryUnitIndex │
   * ├─────────────────┼──────────────────┼─────────────────┤
   * │ 8 bits          │ 14 bits          │ 10 bits         │
   * └─────────────────┴──────────────────┴─────────────────┘
   */

  static constexpr uint32_t MemoryUnitIndexMask = 0x000003FFu;
  static constexpr uint32_t MemoryBlockIndexMask = 0x00FFFC00u;
  static constexpr uint32_t SuperBlockIndexMask = 0xFF000000u;

  static constexpr uint32_t MemoryBlockIndexOffset = 10;
  static constexpr uint32_t SuperBlockIndexOffset = 24;

public:
  __device__ __host__ __forceinline__ uint32_t
  getSuperBlockIndex(SlabAllocAddressT Address) const {
    return Address >> SuperBlockIndexOffset;
  }

  __device__ __host__ __forceinline__ uint32_t
  getMemBlockIndex(SlabAllocAddressT Address) const {
    return (Address & MemoryBlockIndexMask) >> MemoryBlockIndexOffset;
  }

  __device__ __host__ __forceinline__ uint32_t
  getMemUnitIndex(SlabAllocAddressT Address) const {
    return Address & MemoryUnitIndexMask;
  }

  __device__ __forceinline__ uint32_t *
  getPointerFromSlab(const SlabAllocAddressT &Addr, const uint32_t &LaneID) {
    uint32_t SuperBlockIndex = getSuperBlockIndex(Addr);
    uint32_t MemoryBlockIndex = getMemBlockIndex(Addr);
    uint32_t MemoryUnitIndex = getMemUnitIndex(Addr);

    SuperBlock *TheSuperBlock = reinterpret_cast<SuperBlock *>(
        SuperBlocks[SBAllocOffset + SuperBlockIndex]);
    return reinterpret_cast<uint32_t *>(
               &TheSuperBlock
                    ->TheMemoryBlocks[MemoryBlockIndex][MemoryUnitIndex]) +
           LaneID;
  }

  __device__ __forceinline__ uint32_t *
  getPointerForBitmap(const uint32_t SuperBlockIndex,
                      const uint32_t BitMapIndex) {
    SuperBlock *SBPtr = SuperBlocks[SBAllocOffset + SuperBlockIndex];
    return reinterpret_cast<uint32_t *>(&SBPtr->TheBitMap[BitMapIndex]);
  }

  __device__ __forceinline__ void createMemBlockIndex(uint32_t GlobalWarpID) {
    SuperBlockIndex = GlobalWarpID % NumberOfSuperBlocks;
    ResidentIndex = (HashCoefficient * GlobalWarpID) >> (32 - MemoryBlocksLogN);
  }

  __device__ __forceinline__ void updateMemBlockIndex(uint32_t GlobalWarpID) {
    uint32_t LaneID = threadIdx.x & 0x1F;

    ++NumberOfAttempts;
    ++SuperBlockIndex;

    SuperBlockIndex =
        (SuperBlockIndex == NumberOfSuperBlocks) ? 0 : SuperBlockIndex;
    ResidentIndex = (HashCoefficient * (NumberOfAttempts + GlobalWarpID)) >>
                    (32 - MemoryBlocksLogN);

    SuperBlock *SBPtr = SuperBlocks[SBAllocOffset + SuperBlockIndex];
    MemoryBlockBitMap &MBBRef = SBPtr->TheBitMap[ResidentIndex];
    ResidentBitMap = MBBRef[LaneID];
  }

  __device__ __forceinline__ void initAllocator(uint32_t &ThreadID,
                                                uint32_t &LaneID) {
    uint32_t GlobalWarpID = (ThreadID >> 5);
    createMemBlockIndex(GlobalWarpID);

    SuperBlock *SBPtr = SuperBlocks[SBAllocOffset + SuperBlockIndex];
    MemoryBlockBitMap &MBBRef = SBPtr->TheBitMap[ResidentIndex];
    ResidentBitMap = MBBRef[LaneID];
  }

  __device__ __forceinline__ uint32_t warpAllocate(const uint32_t &LaneID) {
    uint32_t AllocatedResult = 0xFFFFFFFF;
    int32_t EmptyLane = -1;
    uint32_t FreeLane;
    uint32_t ReadBitMap = ResidentBitMap;
    uint32_t GlobalWarpID = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    while (AllocatedResult == 0xFFFFFFFF) {
      /* Check whether there exists an empty memory unit associated with thread
       */
      EmptyLane = __ffs(~ResidentBitMap) - 1;

      /* Check if threads in warp have empty memory units */
      FreeLane = __ballot_sync(0xFFFFFFFF, EmptyLane >= 0);

      if (FreeLane == 0) {
        updateMemBlockIndex(GlobalWarpID);
        ReadBitMap = ResidentBitMap;
      } else {
        uint32_t SourceLane = __ffs(FreeLane) - 1;
        if (SourceLane == LaneID) {
          SuperBlock *SBPtr = SuperBlocks[SBAllocOffset + SuperBlockIndex];
          MemoryBlockBitMap &MBBRef = SBPtr->TheBitMap[ResidentIndex];

          ReadBitMap = atomicCAS(&MBBRef[LaneID], ResidentBitMap,
                                 ResidentBitMap | (1 << EmptyLane));
          if (ReadBitMap == ResidentBitMap) {
            ResidentBitMap = ResidentBitMap | (1 << EmptyLane);

            /*
             * RATIONALE:
             * - There are 1024 memory units in one memory block.
             * - There are 1024 bits in the MemoryBlockBitMap (typedef for
             *   uint32_t[WarpSize])
             * - Each thread in a warp with LaneID LI is responsible for
             *   MemoryBlockBitMap[LI]
             * - The first thread is responsible for memory units 0-31; the
             *   second thread is responsible for memory units 32-63, and so on.
             *   We shall call such groups of memory units as memory unit
             *   chunks.
             * - The first five bits (2^5 = 32, warp size) used to identify
             *   memory unit chunk. The next five bits are identify the
             *   previously unallocated memory unit within the memory unit
             *   chunk.
             */
            uint32_t MemoryUnitIndex = (LaneID << 5) | EmptyLane;

            AllocatedResult = (SuperBlockIndex << SuperBlockIndexOffset) |
                              (ResidentIndex << MemoryBlockIndexOffset) |
                              MemoryUnitIndex;
          } else {
            ResidentBitMap = ReadBitMap;
          }
        }
        AllocatedResult = __shfl_sync(0xFFFFFFFF, AllocatedResult, SourceLane);
      }
    }

    return AllocatedResult;
  }

  __device__ __forceinline__ uint32_t warpAllocateBulk(const uint32_t &LaneID,
                                                       const uint32_t N) {
    /*
     * TODO: Implement warpAllocateBulk
     * PRIORITY: Low
     */
  }

  __device__ __forceinline__ void freeUntouched(SlabAllocAddressT Ptr) {
    uint32_t SuperBlockIndex = getSuperBlockIndex(Ptr);
    uint32_t MemoryBlockIndex = getMemBlockIndex(Ptr);
    uint32_t MemoryUnitIndex = getMemUnitIndex(Ptr);

    SuperBlock *SBPtr = SuperBlocks[SBAllocOffset + SuperBlockIndex];
    MemoryBlockBitMap &MBBMRef = SBPtr->TheBitMap[MemoryBlockIndex];

    atomicAnd(&MBBMRef[MemoryUnitIndex >> 5], ~((1 << MemoryUnitIndex) & 0x1F));
  }

  __device__ __host__ __forceinline__ void
  printAddress(SlabAllocAddressT Addr) {
    printf("Super Block Index: %d, "
           "Memory Block Index: %d, "
           "Memory Unit Index: %d"
           "\n",
           getSuperBlockIndex(Addr), getMemBlockIndex(Addr),
           getMemUnitIndex(Addr));
  }

  __device__ __forceinline__ void debug() {
    printf("[SlabAllocContext] Thread: %d, Hash Coefficient: %u, "
           "Number of "
           "Attempts: %u, Resident Index: %u, Resident BitMap: %x, "
           "SuperBlock "
           "Index: %u\n",
           threadIdx.x, HashCoefficient, NumberOfAttempts, ResidentIndex,
           ResidentBitMap, SuperBlockIndex);
  }

private:
  friend class SlabAlloc<MemoryBlocksLogN, SuperBlocksN, WordsPerMemoryUnit>;

  uint32_t HashCoefficient;
  uint32_t NumberOfAttempts;
  uint32_t ResidentIndex;
  uint32_t ResidentBitMap;
  uint32_t SuperBlockIndex;
  uint32_t SBAllocOffset;
};

template <uint32_t MemoryBlocksLogN, uint32_t SuperBlocksN,
          uint32_t WordsPerMemoryUnit = 1>
class SlabAlloc {
public:
  using AllocContext =
      SlabAllocContext<MemoryBlocksLogN, SuperBlocksN, WordsPerMemoryUnit>;
  using SuperBlockTy = typename AllocContext::SuperBlock;

  SlabAlloc() : CleanupCommands{}, TheSlabAllocContext{} {
    std::mt19937 RandomNumberGenerator{
        static_cast<unsigned long>(std::time(0))};
    uint32_t HashCoefficient = RandomNumberGenerator();
    Executor<true> BlockSetup;

    assert(
        ((MAX_SUPERBLOCK_ALLOCATIONS - NextAllocationOffset) >= SuperBlocksN) &&
        "Cannot allocate requested number of superblocks");

    uint32_t SBAllocOffset = NextAllocationOffset;
    NextAllocationOffset += SuperBlocksN;

    for (int Counter = 0; Counter < SuperBlocksN; ++Counter) {
      CHECK_ERROR(cudaMalloc(reinterpret_cast<void **>(&SuperBlocks[Counter]),
                             sizeof(SuperBlockTy)));

      BlockSetup.AddTask(
          [](SuperBlockTy *TheSuperBlock) -> void {
            CHECK_ERROR(cudaMemset(TheSuperBlock->TheBitMap, 0x00,
                                   sizeof(typename AllocContext::BitMap)));
            CHECK_ERROR(
                cudaMemset(TheSuperBlock->TheMemoryBlocks, 0xFF,
                           sizeof(typename AllocContext::MemoryBlocks)));
          },
          TheSuperBlocks[Counter]);

      CleanupCommands.AddTask(
          [](SuperBlockTy *TheSuperBlock) -> void {
            CHECK_ERROR(cudaFree(TheSuperBlock));
          },
          TheSuperBlocks[Counter]);
    }

    BlockSetup.ExecuteTasks();
    TheSlabAllocContext.HashCoefficient = HashCoefficient;
    TheSlabAllocContext.SBAllocOffset = SBAllocOffset;
    cudaMemcpyToSymbol(
        SuperBlocks, TheSuperBlocks, sizeof(SuperBlockTy *) * SuperBlocksN,
        sizeof(SuperBlockTy *) * SBAllocOffset, cudaMemcpyHostToDevice);
  }

  ~SlabAlloc() { CleanupCommands.ExecuteTasks(); };

  AllocContext *getContextPtr() { return &TheSlabAllocContext; }

private:
  Executor<false> CleanupCommands;
  AllocContext TheSlabAllocContext;
  SuperBlockTy *TheSuperBlocks[SuperBlocksN];
};

template <uint32_t LogNumMemoryBlocks, uint32_t NumSuperBlocks,
          uint32_t NumReplicas = 1u>
struct LightAllocatorPolicy {
  static constexpr uint32_t LogNumberOfMemoryBlocks = LogNumMemoryBlocks;
  static constexpr uint32_t NumberOfSuperBlocks = NumSuperBlocks;
  static constexpr uint32_t NumberOfReplicas = NumReplicas;

  using DynamicAllocatorT =
      SlabAllocLight<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
  using AllocatorContextT =
      SlabAllocLightContext<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
};

template <uint32_t LogNumMemoryBlocks, uint32_t NumSuperBlocks,
          uint32_t NumReplicas = 1u>
struct FullAllocatorPolicy {
  static constexpr uint32_t LogNumberOfMemoryBlocks = LogNumMemoryBlocks;
  static constexpr uint32_t NumberOfSuperBlocks = NumSuperBlocks;
  static constexpr uint32_t NumberOfReplicas = NumReplicas;

  using DynamicAllocatorT =
      SlabAlloc<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
  using AllocatorContextT =
      SlabAllocLight<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
};
