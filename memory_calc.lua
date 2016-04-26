
local Fsize_bytes ,Zsize_bytes = 4,8

local K = 50
local Np = 8
local No = 2
local M = 1280
local Nx = 2500
local Ny = 2500

local bytes_per_MB = 2^20
local bytes_per_GB = 2^30

local O_mem = No * Nx * Ny * Zsize_bytes / bytes_per_MB
local O_denom_mem = No * Nx * Ny * Fsize_bytes / bytes_per_MB
local P_mem = Np * M * M * Zsize_bytes / bytes_per_MB
local Z_mem = 3 * K * No * Np * M * M * Zsize_bytes / bytes_per_MB
local Z_mem_GB = 3 * K * No * Np * M * M * Zsize_bytes / bytes_per_GB
local total = O_mem + P_mem + O_denom_mem + Z_mem
print('O_mem = '..O_mem..' MB')
print('O_denom_mem = '..O_denom_mem..' MB')
print('P_mem = '..P_mem..' MB')
print('Z_mem = '..Z_mem_GB..' GB')
print('total = '..total..' MB')
