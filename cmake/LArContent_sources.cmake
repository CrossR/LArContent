# Dynamically collect all .cc files in the larpandoracontent directory and subdirectories
file(GLOB_RECURSE LAR_CONTENT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/larpandoracontent/*.cc
)
