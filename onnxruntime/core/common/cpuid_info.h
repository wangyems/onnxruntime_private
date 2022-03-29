// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

#if defined(_M_IX86) || (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__i386__) || defined(__x86_64__)
#define CPUIDINFO_ARCH_X86
#endif

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#define CPUIDINFO_ARCH_ARM
#endif  // ARM or ARM64

namespace onnxruntime {

class CPUIDInfo {
 public:
  static const CPUIDInfo& GetCPUIDInfo() {
    static CPUIDInfo cpuid_info;
    return cpuid_info;
  }

  bool HasAVX() const { return has_avx_; }
  bool HasAVX2() const { return has_avx2_; }
  bool HasAVX512f() const { return has_avx512f_; }
  bool HasAVX512Skylake() const { return has_avx512_skylake_; }
  bool HasF16C() const { return has_f16c_; }
  bool HasSSE3() const { return has_sse3_; }
  bool HasSSE4_1() const { return has_sse4_1_; }
  bool IsHybrid() const { return is_hybrid_; }

  // ARM 
  bool HasArmNeonDot() const { return has_arm_neon_dot_; }

  /**
   * @return CPU core micro-architecture running the current thread
  */
  int32_t GetCurrentUarch() const;

  /**
   * @return CPU core micro-architecture
  */
  int32_t GetCoreUarch(uint32_t coreId) const;

  /**
  * @brief Some ARMv8 power efficient core has narrower 64b load/store
  *        that needs specialized optimiztion in kernels
  * @return whether the indicated core has narrower load/store device
  */
  bool IsCoreArmv8NarrowLd(uint32_t coreId) const;
  
  /**
  * @brief Some ARMv8 power efficient core has narrower 64b load/store
  *        that needs specialized optimiztion in kernels
  * @return whether the current core has narrower load/store device
  */
  bool IsCurrentCoreArmv8NarrowLd() const;

 private:
  CPUIDInfo();
  bool has_avx_{false};
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_avx512_skylake_{false};
  bool has_f16c_{false};
  bool has_sse3_{false};
  bool has_sse4_1_{false};
  bool is_hybrid_{false};

  std::vector<uint32_t> core_uarchs_; // micro-arch of each core

  // In ARMv8 systems, some power efficient cores has narrower
  // 64b load/store devices. It takes longer for them to load
  // 128b vectore registers.
  std::vector<bool> is_armv8_narrow_ld_;


#if (defined(CPUIDINFO_ARCH_X86) || defined(CPUIDINFO_ARCH_ARM)) && defined(CPUINFO_SUPPORTED)
  bool pytorch_cpuinfo_init_{false};
#endif
  bool has_arm_neon_dot_{false};
};

}  // namespace onnxruntime
