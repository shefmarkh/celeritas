#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4 AND CELERITAS_USE_ROOT)
  # Exporter
  add_executable(geant-exporter geant-exporter/geant-exporter.cc
    geant-exporter/ActionInitialization.cc
    geant-exporter/DetectorConstruction.cc
    geant-exporter/PhysicsList.cc
    geant-exporter/PrimaryGeneratorAction.cc
    geant-exporter/GeantPhysicsTableWriter.cc
    # TODO: Move these to main celeritas::geant library for use in acceleritas?
    geant-exporter/GeantExceptionHandler.cc
    geant-exporter/GeantLoggerAdapter.cc
  )
  target_link_libraries(geant-exporter
    Celeritas::Core
    Celeritas::ROOT
    ROOT::Core
    ROOT::Tree
    XercesC::XercesC
    ${Geant4_LIBRARIES}
  )

  # Reader
  add_executable(geant-exporter-cat geant-exporter/geant-exporter-cat.cc)
  target_link_libraries(geant-exporter-cat
    Celeritas::Core
    Celeritas::ROOT
  )

  if(CELERITAS_BUILD_TESTS)
    set(_geant_test_inp
      "${CMAKE_CURRENT_SOURCE_DIR}/geant-exporter/data/four-steel-slabs.gdml"
    )
    add_test(NAME "app/geant-exporter"
      COMMAND "$<TARGET_FILE:geant-exporter>"
      "${_geant_test_inp}" "test-data.root"
    )

    # Set environment variables from Geant4-exported configuration
    set(_geant_test_env)
    foreach(_ds IN LISTS Geant4_DATASETS)
      list(APPEND _geant_test_env
        "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}")
    endforeach()
    set_tests_properties("app/geant-exporter" PROPERTIES
      ENVIRONMENT "${_geant_test_env}"
      REQUIRED_FILES "${_geant_test_inp}"
    )

    add_test(NAME "app/geant-exporter-cat"
      COMMAND "$<TARGET_FILE:geant-exporter-cat>"
        "test-data.root"
    )
    set_tests_properties("app/geant-exporter-cat" PROPERTIES
      DEPENDS "app/geant-exporter"
      REQUIRED_FILES "test-data.root"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: physics interactions
#-----------------------------------------------------------------------------#
if(CELERITAS_BUILD_DEMOS AND NOT CELERITAS_USE_JSON)
  message(SEND_ERROR "JSON support is required for demos")
endif()

if(CELERITAS_BUILD_DEMOS)
  add_library(celeritas_demo_interactor
    demo-interactor/LoadXs.cc
    demo-interactor/KNDemoIO.cc
    demo-interactor/XsGridParams.cc
  )
  target_link_libraries(celeritas_demo_interactor PUBLIC
    celeritas
    nlohmann_json::nlohmann_json
  )

  if(CELERITAS_USE_CUDA)
    add_executable(demo-interactor
      demo-interactor/demo-interactor.cc
      demo-interactor/KNDemoRunner.cc
      demo-interactor/KNDemoKernel.cu
      demo-interactor/KNDemoKernel.thrust.cu
    )
    target_link_libraries(demo-interactor celeritas_demo_interactor)

    if(CELERITAS_BUILD_TESTS)
      set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-interactor/simple-driver.py")
      add_test(NAME "app/demo-interactor"
        COMMAND "$<TARGET_FILE:Python::Interpreter>" "${_driver}"
      )
      set(_env
        "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-interactor>"
        "CELER_DISABLE_PARALLEL=1"
      )
      set_tests_properties("app/demo-interactor" PROPERTIES
        ENVIRONMENT "${_env}"
        RESOURCE_LOCK gpu
        REQUIRED_FILES "${_driver}"
      )
    endif()
  endif()

  # Build CPU version
  add_executable(host-demo-interactor
    demo-interactor/host-demo-interactor.cc
    demo-interactor/HostKNDemoRunner.cc
  )
  target_link_libraries(host-demo-interactor
    celeritas
    celeritas_demo_interactor
  )

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-interactor/simple-driver.py")
    add_test(NAME "app/host-demo-interactor"
      COMMAND "$<TARGET_FILE:Python::Interpreter>" "${_driver}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:host-demo-interactor>"
      "CELER_DISABLE_DEVICE=1"
      "CELER_DISABLE_PARALLEL=1"
    )
    set_tests_properties("app/host-demo-interactor" PROPERTIES
      ENVIRONMENT "${_env}"
      REQUIRED_FILES "${_driver}"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: physics interactions with Alpaka
#-----------------------------------------------------------------------------#

find_package(alpaka QUIET)
find_package(nlohmann_json)

if(NOT alpaka_FOUND)
  message(STATUS "No alpaka install found, disabling build of FisherPrice_Alpaka")
else()

  configure_file("../src/celeritas_config.h.in" "celeritas_config.h" @ONLY)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/celeritas_config.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

  configure_file("../src/celeritas_version.cc.in" "celeritas_version.cc" @ONLY)

  alpaka_add_executable(demo-interactor-alpaka demo-interactor-alpaka/demo-interactor-alpaka.cc    
    demo-interactor-alpaka/KNDemoKernel.cu
    demo-interactor-alpaka/KNDemoKernel.thrust.cu
    demo-interactor-alpaka/KNDemoIO.cc
    demo-interactor-alpaka/KNDemoRunner.cc
    demo-interactor-alpaka/LoadXs.cc
    demo-interactor-alpaka/XsGridParams.cc
    ../src/physics/base/ParticleParams.cc
    ../src/comm/DeviceIO.json.cc
    ../src/comm/KernelDiagnosticsIO.json.cc
    ../src/comm/Device.cc
    ../src/base/Assert.cc
    ../src/comm/Logger.cc
    ../src/comm/LoggerTypes.cc
    ../src/comm/detail/LoggerMessage.cc
    ../src/comm/ScopedMpiInit.cc
    ../src/comm/Communicator.cc
    ../src/comm/KernelDiagnostics.cc
    ../src/base/DeviceAllocation.cc
    ../src/base/ColorUtils.cc
    ../src/base/KernelParamCalculator.cuda.cc
    ../src/base/detail/Copier.cc
    ../src/base/detail/Filler.cu
    ../src/random/RngInterface.cc
    ../src/random/detail/RngStateInit.cu
    ${CMAKE_CURRENT_BINARY_DIR}/celeritas_version.cc
  )


  target_include_directories(demo-interactor-alpaka
  PUBLIC    
    "${CMAKE_CURRENT_BINARY_DIR}"
    ../src/
  )


  target_link_libraries(demo-interactor-alpaka PRIVATE 
    CUDA::cudart
    nlohmann_json::nlohmann_json
    alpaka::alpaka
    ${Boost_LIBRARIES}
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
 endif()


#-----------------------------------------------------------------------------#
# DEMO: geometry tracking
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
  # Since the demo kernel links against VecGeom, which requires CUDA separable
  # compilation, it cannot be linked directly into an executable.
  add_library(celeritas_demo_rasterizer
    demo-rasterizer/RDemoRunner.cc
    demo-rasterizer/RDemoKernel.cu
    demo-rasterizer/ImageIO.cc
    demo-rasterizer/ImageStore.cc
  )
  target_link_libraries(celeritas_demo_rasterizer
    PRIVATE
      celeritas
      nlohmann_json::nlohmann_json
  )
  celeritas_link_vecgeom_cuda(celeritas_demo_rasterizer)

  # Add the executable
  add_executable(demo-rasterizer
    demo-rasterizer/demo-rasterizer.cc
  )
  target_link_libraries(demo-rasterizer
    celeritas
    VecGeom::vecgeom
    celeritas_demo_rasterizer
    nlohmann_json::nlohmann_json
  )

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-rasterizer/simple-driver.py")
    set(_gdml_inp "${PROJECT_SOURCE_DIR}/test/geometry/data/twoBoxes.gdml")
    add_test(NAME "app/demo-rasterizer"
      COMMAND "$<TARGET_FILE:Python::Interpreter>" "${_driver}" "${_gdml_inp}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-rasterizer>"
      "CELER_DISABLE_PARALLEL=1"
    )
    set_tests_properties("app/demo-rasterizer" PROPERTIES
      ENVIRONMENT "${_env}"
      RESOURCE_LOCK gpu
      REQUIRED_FILES "${_driver};${_gdml_inp}"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: full physics loop
#-----------------------------------------------------------------------------#
if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_VecGeom)
  set(_cuda_src)
  if(CELERITAS_USE_CUDA)
    set(_cuda_src
      demo-loop/LDemoKernel.cu
    )
  endif()
  add_library(celeritas_demo_loop
    demo-loop/LDemoIO.cc
    demo-loop/LDemoParams.cc
    demo-loop/LDemoRun.cc
    demo-loop/demo-loop.cc
    ${_cuda_src}
  )

  if(CELERITAS_USE_CUDA)
    celeritas_link_vecgeom_cuda(celeritas_demo_loop)
  endif()
  target_link_libraries(celeritas_demo_loop
    PRIVATE
    celeritas
    nlohmann_json::nlohmann_json
    Celeritas::ROOT
  )

  # Add the executable
  add_executable(demo-loop
    demo-loop/demo-loop.cc
  )
  target_link_libraries(demo-loop
    celeritas
    VecGeom::vecgeom
    celeritas_demo_loop
    nlohmann_json::nlohmann_json
  )

endif()

#-----------------------------------------------------------------------------#
