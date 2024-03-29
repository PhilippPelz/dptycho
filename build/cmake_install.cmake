# Install script for directory: /home/philipp/projects/dptycho

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
    "/home/philipp/projects/dptycho/2015-11-25.lua"
    "/home/philipp/projects/dptycho/2015-11-25_scan5.lua"
    "/home/philipp/projects/dptycho/2016-03-14-scan1.lua"
    "/home/philipp/projects/dptycho/2016-03-14.lua"
    "/home/philipp/projects/dptycho/2016-04-11.lua"
    "/home/philipp/projects/dptycho/2016-05-09.lua"
    "/home/philipp/projects/dptycho/2016-05-17.lua"
    "/home/philipp/projects/dptycho/2016-05-26.lua"
    "/home/philipp/projects/dptycho/2017-01-25_carbon2_scan1.lua"
    "/home/philipp/projects/dptycho/TWF_simul.lua"
    "/home/philipp/projects/dptycho/init.lua"
    "/home/philipp/projects/dptycho/main.lua"
    "/home/philipp/projects/dptycho/main2.lua"
    "/home/philipp/projects/dptycho/memory_calc.lua"
    "/home/philipp/projects/dptycho/pca.lua"
    "/home/philipp/projects/dptycho/ptycho.lua"
    "/home/philipp/projects/dptycho/ptycho2.lua"
    "/home/philipp/projects/dptycho/ptycho20.lua"
    "/home/philipp/projects/dptycho/ptycho2twf.lua"
    "/home/philipp/projects/dptycho/ptycho_bg.lua"
    "/home/philipp/projects/dptycho/ptycho_bg_generate.lua"
    "/home/philipp/projects/dptycho/ptycho_nosubpix.lua"
    "/home/philipp/projects/dptycho/ptycho_old.lua"
    "/home/philipp/projects/dptycho/ptycho_poisson.lua"
    "/home/philipp/projects/dptycho/ptycho_poisson_generate.lua"
    "/home/philipp/projects/dptycho/ptycho_runner.lua"
    "/home/philipp/projects/dptycho/ptyrecon.lua"
    "/home/philipp/projects/dptycho/ptysimul.lua"
    "/home/philipp/projects/dptycho/test_shift.lua"
    "/home/philipp/projects/dptycho/test/test.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/core" TYPE FILE FILES
    "/home/philipp/projects/dptycho/core/init.lua"
    "/home/philipp/projects/dptycho/core/netbuilder.lua"
    "/home/philipp/projects/dptycho/core/potential.lua"
    "/home/philipp/projects/dptycho/core/propagators.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/core/ptycho" TYPE FILE FILES
    "/home/philipp/projects/dptycho/core/ptycho/DM_engine.lua"
    "/home/philipp/projects/dptycho/core/ptycho/RAAR_engine.lua"
    "/home/philipp/projects/dptycho/core/ptycho/RWF_engine.lua"
    "/home/philipp/projects/dptycho/core/ptycho/Runner.lua"
    "/home/philipp/projects/dptycho/core/ptycho/TWF_engine.lua"
    "/home/philipp/projects/dptycho/core/ptycho/base_engine.lua"
    "/home/philipp/projects/dptycho/core/ptycho/init.lua"
    "/home/philipp/projects/dptycho/core/ptycho/initialization.lua"
    "/home/philipp/projects/dptycho/core/ptycho/ops.lua"
    "/home/philipp/projects/dptycho/core/ptycho/ops_general.lua"
    "/home/philipp/projects/dptycho/core/ptycho/ops_subpixel.lua"
    "/home/philipp/projects/dptycho/core/ptycho/params.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/io" TYPE FILE FILES
    "/home/philipp/projects/dptycho/io/dataloader.lua"
    "/home/philipp/projects/dptycho/io/init.lua"
    "/home/philipp/projects/dptycho/io/plot.lua"
    "/home/philipp/projects/dptycho/io/reconstruction_plot.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/util" TYPE FILE FILES
    "/home/philipp/projects/dptycho/util/allocator.lua"
    "/home/philipp/projects/dptycho/util/init.lua"
    "/home/philipp/projects/dptycho/util/linear_schedule.lua"
    "/home/philipp/projects/dptycho/util/physics.lua"
    "/home/philipp/projects/dptycho/util/stats.lua"
    "/home/philipp/projects/dptycho/util/tabular_schedule.lua"
    "/home/philipp/projects/dptycho/util/stats.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/util" TYPE FILE FILES "/home/philipp/projects/dptycho/util/stats.py")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/znn" TYPE FILE FILES
    "/home/philipp/projects/dptycho/znn/AddConst.lua"
    "/home/philipp/projects/dptycho/znn/AtomRadiusPenalty.lua"
    "/home/philipp/projects/dptycho/znn/BM3D_MSE_Criterion.lua"
    "/home/philipp/projects/dptycho/znn/BM3D_MSE_Criterion2.lua"
    "/home/philipp/projects/dptycho/znn/CMul.lua"
    "/home/philipp/projects/dptycho/znn/CMulModule.lua"
    "/home/philipp/projects/dptycho/znn/ComplexAbs.lua"
    "/home/philipp/projects/dptycho/znn/ConvFFT2D.lua"
    "/home/philipp/projects/dptycho/znn/ConvParams.lua"
    "/home/philipp/projects/dptycho/znn/ConvSlice.lua"
    "/home/philipp/projects/dptycho/znn/EuclideanLoss.lua"
    "/home/philipp/projects/dptycho/znn/FFT.lua"
    "/home/philipp/projects/dptycho/znn/MultiCriterionVariableWeights.lua"
    "/home/philipp/projects/dptycho/znn/PoissonLikelihood.lua"
    "/home/philipp/projects/dptycho/znn/Select.lua"
    "/home/philipp/projects/dptycho/znn/Source.lua"
    "/home/philipp/projects/dptycho/znn/SpatialSmoothnessCriterion.lua"
    "/home/philipp/projects/dptycho/znn/Sqrt.lua"
    "/home/philipp/projects/dptycho/znn/SqrtInPlace.lua"
    "/home/philipp/projects/dptycho/znn/Square.lua"
    "/home/philipp/projects/dptycho/znn/Sum.lua"
    "/home/philipp/projects/dptycho/znn/SupportMask.lua"
    "/home/philipp/projects/dptycho/znn/THZNN.lua"
    "/home/philipp/projects/dptycho/znn/TVCriterion.lua"
    "/home/philipp/projects/dptycho/znn/Threshold.lua"
    "/home/philipp/projects/dptycho/znn/TraceCriterion.lua"
    "/home/philipp/projects/dptycho/znn/TracePenalty.lua"
    "/home/philipp/projects/dptycho/znn/TruncatedPoissonLikelihood.lua"
    "/home/philipp/projects/dptycho/znn/VolumetricConvolutionFixedFilter.lua"
    "/home/philipp/projects/dptycho/znn/WSECriterion.lua"
    "/home/philipp/projects/dptycho/znn/WeightedL1Cost.lua"
    "/home/philipp/projects/dptycho/znn/WeightedLinearCriterion.lua"
    "/home/philipp/projects/dptycho/znn/init.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/simulation" TYPE FILE FILES
    "/home/philipp/projects/dptycho/simulation/init.lua"
    "/home/philipp/projects/dptycho/simulation/simulator.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/simulation" TYPE FILE FILES
    "/home/philipp/projects/dptycho/simulation/mtfdqe.py"
    "/home/philipp/projects/dptycho/simulation/probe.py"
    "/home/philipp/projects/dptycho/simulation/random_probe.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/scripts" TYPE FILE FILES
    "/home/philipp/projects/dptycho/scripts/experiment_suite.lua"
    "/home/philipp/projects/dptycho/scripts/experiment_suite_DM.lua"
    "/home/philipp/projects/dptycho/scripts/figure_averaging.lua"
    "/home/philipp/projects/dptycho/scripts/prepare_probes.lua"
    "/home/philipp/projects/dptycho/scripts/ptycho_runner.lua"
    "/home/philipp/projects/dptycho/scripts/test_simulation.lua"
    "/home/philipp/projects/dptycho/scripts/test_simulation0.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/dptycho/scm-1/lua/dptycho/test" TYPE FILE FILES "/home/philipp/projects/dptycho/test/test.lua")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/philipp/projects/dptycho/build/znn/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/philipp/projects/dptycho/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
