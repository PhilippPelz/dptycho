# Install script for directory: /media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/philipp/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2015-11-25.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2015-11-25_scan5.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2016-03-14-scan1.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2016-03-14.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2016-04-11.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2016-05-09.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2016-05-17.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/2016-05-26.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/TWF_simul.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/init.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/main.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/main2.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/memory_calc.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/pca.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho2.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_bg.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_bg_generate.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_nosubpix.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_old.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_poisson.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_poisson_generate.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptycho_runner.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptyrecon.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/ptysimul.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/test_shift.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/test/test.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/core" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/init.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/netbuilder.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/potential.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/propagators.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/core/ptycho" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/BM3D_TWF_engine.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/DM_engine.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/DM_engine_subpix.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/RAAR_engine.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/RWF_engine.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/Runner.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/TWF_engine.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/TWF_engine_subpix.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/base_engine.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/base_engine_shifted.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/init.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/initialization.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/ops.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/ops_general.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/ops_subpixel.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/core/ptycho/params.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/io" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/io/dataloader.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/io/init.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/io/plot.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/io/reconstruction_plot.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/util" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/allocator.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/init.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/linear_schedule.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/physics.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/stats.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/tabular_schedule.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/stats.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/util" TYPE FILE FILES "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/util/stats.py")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/znn" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/AddConst.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/AtomRadiusPenalty.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/BM3D_MSE_Criterion.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/BM3D_MSE_Criterion2.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/CMul.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/CMulModule.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/ComplexAbs.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/ConvFFT2D.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/ConvParams.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/ConvSlice.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/EuclideanLoss.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/FFT.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/MultiCriterionVariableWeights.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/PoissonLikelihood.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/Select.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/Source.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/SpatialSmoothnessCriterion.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/Sqrt.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/SqrtInPlace.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/Square.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/Sum.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/SupportMask.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/THZNN.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/TVCriterion.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/Threshold.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/TraceCriterion.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/TracePenalty.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/TruncatedPoissonLikelihood.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/VolumetricConvolutionFixedFilter.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/WSECriterion.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/WeightedL1Cost.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/WeightedLinearCriterion.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/znn/init.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/simulation" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/simulation/init.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/simulation/simulator.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/simulation" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/simulation/mtfdqe.py"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/simulation/probe.py"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/simulation/random_probe.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/scripts" TYPE FILE FILES
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/experiment_suite.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/experiment_suite_DM.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/figure_averaging.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/prepare_probes.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/ptycho_runner.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/test_simulation.lua"
    "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/scripts/test_simulation0.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/test" TYPE FILE FILES "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/test/test.lua")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/build/znn/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/media/philipp/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/home/philipp/projects/dptycho/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
