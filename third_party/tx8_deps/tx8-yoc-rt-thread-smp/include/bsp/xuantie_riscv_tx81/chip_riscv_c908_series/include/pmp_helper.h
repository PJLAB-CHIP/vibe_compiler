//
// Created by wangribin on 2026/1/9.
//

#ifndef TX81FW_PMP_HELPER_H
#define TX81FW_PMP_HELPER_H
#include <stdint.h>
#include <stdbool.h>

#include "riscv_csr.h"
#include "core/csi_rv_common.h"

// 定义 PMP 配置位的掩码
#define PMP_R     0x01
#define PMP_W     0x02
#define PMP_X     0x04
#define PMP_A_OFF   0x00
#define PMP_A_TOR   0x08
#define PMP_A_NA4   0x10
#define PMP_A_NAPOT 0x18
#define PMP_L     0x80
#define MINI_PMP_ADDR_SIZE     0x1000

#define MXX_PMP_ADDR_NUMBER     64



/**
  \brief   Set PMPxCFG by index
  \details Writes the given value to the PMPxCFG Register.
  \param [in]    idx      PMPx region index
  \param [in]    pmpxcfg  PMPxCFG Register value to set
 */
void set_PMPxCFG(unsigned long idx, uint8_t pmpxcfg);

/**
 * 检查整数是否为 2 的幂次 (Power of 2)
 */
static inline bool is_power_of_two(uint64_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

/**
 * @brief TOR 模式只需要将物理地址右移 2 位即可
 * @param phys_addr 物理地址
 * @return 写入 pmpaddr 寄存器的值
 */
static inline uint64_t pmp_calc_tor_val(uint64_t phys_addr) {
    return phys_addr >> 2;
}

/**
 * @brief 计算 NAPOT 模式下的 pmpaddr 寄存器值
 * * RISC-V NAPOT 编码逻辑：
 * pmpaddr = (base_addr >> 2) | ((size >> 3) - 1)
 * * @param phys_addr 物理基地址 (必须按照 size 对齐)
 * @param size      区域大小 (必须是 2 的幂次，且 >= 8 字节)
 * @return uint64_t 写入 pmpaddr 寄存器的编码值。如果参数非法返回 0。
 */
inline static  uint64_t pmp_calc_napot_addr(uint64_t phys_addr, uint64_t size) {
    // 1. 安全检查：大小必须是 2 的幂次
    if (!is_power_of_two(size)) {
        return 0; // 错误：Size 非 2 的幂
    }

    // 2. 安全检查：地址必须按照大小自然对齐
    if ((phys_addr & (size - 1)) != 0) {
        return 0; // 错误：地址未对齐
    }

    // 3. 安全检查：NAPOT 要求最小 4K 字节 (RV64)
    if (size < MINI_PMP_ADDR_SIZE) {
        return 0;
    }

    /* * 核心算法解释：
     * 假设我们要保护 8KB (0x2000)。
     * size >> 1 (0x1000) -> size/2
     * size/2 - 1 (0xFFF) -> 创建低位全是 1 的掩码，长度对应 (order-1)
     * base | mask -> 将掩码应用到基地址低位
     * >> 2 -> 因为 pmpaddr 实际上编码的是 bit[55:2]
     */
    uint64_t pmpaddr_val = (phys_addr | ((size >> 1) - 1)) >> 2;

    return pmpaddr_val;
}

/**
 * @brief 构建 pmpcfg 配置字节
 * * @param mode  PMP_A_NAPOT or PMP_A_TOR
 * @param r  允许读取
 * @param w  允许写入
 * @param x  允许执行
 * @param l  锁定 (Lock) - 一旦置位，直到复位前不可更改，且规则对 M-Mode 生效
 * @return uint8_t
 */
inline static  uint8_t pmp_build_cfg(uint8_t mode,bool r, bool w, bool x, bool l) {
    uint8_t cfg = mode;

    if (r) cfg |= PMP_R;
    if (w) cfg |= PMP_W;
    if (x) cfg |= PMP_X;
    if (l) cfg |= PMP_L;

    return cfg;
}
/* ============================================================
 * 工具函数 1: 动态读取 CSR
 * 支持: pmpaddr0-63, pmpcfg0-14 (偶数)
 * ============================================================ */
uint64_t pmp_csr_read_dynamic(uint16_t csr_addr) ;
/* ============================================================
 * 工具函数 2: 动态写入 CSR
 * ============================================================ */
void pmp_csr_write_dynamic(uint16_t csr_addr, uint64_t val);
extern void syn_pmp_addr_idx_for_smp();
extern void set_rest_pmp_item_to_dynamic_pmp_addr_for_so();


extern  bool krt_pmp_init_addr(uint64_t phys_addr, uint64_t size,bool r, bool w, bool x) ;
extern  void krt_clean_all_pmp_config_in_so();

#endif //TX81FW_PMP_HELPER_H
