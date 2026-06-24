include_guard(GLOBAL)

if(NOT DEFINED SOC_RTT_TOOLS_DIR)
    set(SOC_RTT_TOOLS_DIR "${TX8FW_BASE}/tool")
endif ()

set(PRE_LINK_TOOL "${SOC_RTT_TOOLS_DIR}/prelink-riscv")
set(BIN2HEADER_TOOL "${SOC_RTT_TOOLS_DIR}/bin2header")

if (CASENAME)
    set(CASENAME_STRING ${BUILD_STRING}-"CASE."${CASENAME})
else ()
    set(CASENAME_STRING ${BUILD_STRING})
endif ()

if (STREAMCONFIG)
    set(STREAM_STRING ${CASENAME_STRING}-"S_CFG."${STREAMCONFIG})
else ()
    set(STREAM_STRING ${CASENAME_STRING})
endif ()

if (TILEID)
    set(DEFINES  ${DEFINES}BUILD_TILE_ID=\"${TILEID}\")
    set(VERSION_DESCRIBE_STRING ${STREAM_STRING}-"TILE"${TILEID})
else ()
    set(VERSION_DESCRIBE_STRING ${STREAM_STRING})
endif ()

set(TARGET_CPU "c908")
set(ARCH "riscv")
if (NOT DEFINED ADD_BUILD_TYPE_FLAGS)
    # 保证只增加一次
    set(ADD_BUILD_TYPE_FLAGS TRUE)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O2)
        # add_compile_definitions(OUTPUT_DEBUG_INFO=1) 代码接口已变更，适配后开启
        add_compile_definitions(LOG_ON=0)
    else()
        add_compile_options(-O2)

        # 移除 -O3（这里假设 -O3 是单独出现的，不是嵌入在其他选项中的）
        string(REGEX REPLACE "-O3" "" MODIFIED_ASM_FLAGS_RELEASE "${CMAKE_ASM_FLAGS_RELEASE}")
        # 设置修改后的 CMAKE_C_FLAGS_RELEASE
        set(CMAKE_ASM_FLAGS_RELEASE "${MODIFIED_ASM_FLAGS_RELEASE} -O2") # 或者你想要的其他优化级别

        string(REGEX REPLACE "-O3" "" MODIFIED_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        # 设置修改后的 CMAKE_C_FLAGS_RELEASE
        set(CMAKE_C_FLAGS_RELEASE "${MODIFIED_C_FLAGS_RELEASE} -O2") # 或者你想要的其他优化级别

        # 移除 -O3（这里假设 -O3 是单独出现的，不是嵌入在其他选项中的）
        string(REGEX REPLACE "-O3" "" MODIFIED_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        # 设置修改后的 CMAKE_CXX_FLAGS_RELEASE
        set(CMAKE_CXX_FLAGS_RELEASE "${MODIFIED_CXX_FLAGS_RELEASE} -O2") # 或者你想要的其他优化级别

        add_compile_definitions(LOG_ON=0)
    endif()
endif()
message(STATUS "CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")


