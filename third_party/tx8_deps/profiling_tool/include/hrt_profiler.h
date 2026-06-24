/*
 * Copyright (C) 2024 Tsing Micro Intelligent Technology Co.,Ltd. All rights
 * reserved.
 *
 * This file is the property of Tsing Micro Intelligent Technology Co.,Ltd. This
 * file may only be distributed to: (i) a Tsing Micro party having a legitimate
 * business need for the information contained herein, or (ii) a non-Tsing Micro
 * party having a legitimate business need for the information contained herein.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */
#ifndef HRT_PROFILER_H
#define HRT_PROFILER_H

#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

#include "tx_runtime.h"

#define CHIP_MAX_NUM 32
#define TILE_MAX_NUM 16

typedef enum TxProfAction { PROF_START, PROF_STOP } TxProfAction;

/**
 * @brief 下发性能采集命令
 * @return void
 */
void TsmProcessProfData(uint32_t chip_id, std::string graph_name, TxProfAction prof_action, uint16_t prof_type);
#endif