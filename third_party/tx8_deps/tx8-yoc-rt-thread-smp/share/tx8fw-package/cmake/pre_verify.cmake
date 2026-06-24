include(${CMAKE_CURRENT_LIST_DIR}/sdk_info.cmake)
list(LENGTH TX8FW_FIND_COMPONENTS components_length)
set(TX8FW_PRODUCT_SERIES_FOUND OFF)
foreach(__component ${TX8FW_FIND_COMPONENTS})
    if(TX8FW_PRODUCT_SERIES  STREQUAL __component)
        list(REMOVE_ITEM TX8FW_FIND_COMPONENTS ${TX8FW_PRODUCT_SERIES})
        set(TX8FW_PRODUCT_SERIES_FOUND ON)
        return()
    endif ()
endforeach()



