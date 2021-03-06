find_package(Boost COMPONENTS serialization filesystem system python REQUIRED)

add_executable(binarize binarize.cpp )
add_executable(binarize_v2 binarize_v2.cpp )
add_executable(batch_query batch_query.cpp )
add_executable(batch_query_v2 batch_query_v2.cpp )
add_executable(interactive_query interactive_query.cpp )
add_executable(interactive_query_v2 interactive_query_v2.cpp )

target_link_libraries(binarize
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                     )
                     
target_link_libraries(binarize_v2
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                     )

target_link_libraries(batch_query
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      memory_management
                      gpu_search
                     )
                     
target_link_libraries(batch_query_v2
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      memory_management
                      gpu_search_v2
                     )
                     
target_link_libraries(interactive_query
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      memory_management
                      gpu_search
                     )
                     
target_link_libraries(interactive_query_v2
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      memory_management
                      gpu_search_v2
                     )
                     
if (DEFINED PYTHON_INCLUDE_DIR)
    set(Python_ADDITIONAL_VERSIONS ${PYTHON_VER_FLAG})
    find_package(PythonLibs)
    if (${PYTHONLIBS_FOUND})
        add_executable(ngrams4nemantus ngrams4nemantus.cpp )
        target_include_directories(ngrams4nemantus PUBLIC ${PYTHON_INCLUDE_DIR})
        target_link_libraries(ngrams4nemantus
                        ${PYTHON_LIBRARIES} 
                        ${Boost_LIBRARIES}
                        memory_management
                        gpu_search_v2
                        )
    endif()
endif()
