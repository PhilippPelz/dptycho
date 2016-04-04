
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)

SET(package "dptycho")

FILE(GLOB luasrc *.lua)
FILE(GLOB coresrc core/*.lua)
FILE(GLOB coreptychosrc core/ptycho/*.lua)
FILE(GLOB iosrc io/*.lua)
FILE(GLOB utilsrc util/*.lua)
FILE(GLOB znnsrc znn/*.lua)

ADD_SUBDIRECTORY(znn)

SET(luasrc ${luasrc} test/test.lua)
ADD_TORCH_PACKAGE(dptycho "" "${luasrc}")
# INSTALL(FILES ${coresrc} core/libslicelib.so DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package}/core)
INSTALL(FILES ${coresrc} DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package}/core)
INSTALL(FILES ${coreptychosrc} DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package}/core/ptycho)
INSTALL(FILES ${iosrc} DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package}/io)
INSTALL(FILES ${utilsrc} DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package}/util)
INSTALL(FILES ${znnsrc} DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package}/znn)
