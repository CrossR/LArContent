find_ups_product( libtorch )
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)
find_package(TorchSparse REQUIRED)
find_package(TorchScatter REQUIRED)

include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})
include_directories(SYSTEM ${TorchSparse_INCLUDE_DIR})
include_directories(SYSTEM ${TorchScatter_INCLUDE_DIR})

cet_set_compiler_flags(DIAGS CAUTIOUS
      WERROR
      NO_UNDEFINED
      EXTRA_FLAGS
      -pedantic
      ${TORCH_CXX_FLAGS})

set( subdir_list LArControlFlow
                 LArHelpers
                 LArMonitoring
                 LArTrackShowerId
                 LArShowerGrowing
)

cet_make( LIBRARY_NAME LArPandoraDLContent
    SUBDIRS ${subdir_list}
    LIBRARIES
        ${PANDORASDK}
        ${PANDORAMONITORING}
        ${LAR_CONTENT_LIBRARY_NAME}
        ${TORCH_INSTALL_PREFIX}/lib/libtorch.so
        ${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so
        ${TORCH_INSTALL_PREFIX}/lib/libc10.so
)

install_source( SUBDIRS ${subdir_list} )
install_headers( SUBDIRS ${subdir_list} )
