cmake_minimum_required(VERSION 3.8)
project(VRP)

# Spécifier la version de C++
set(CMAKE_CXX_STANDARD 11)

# Inclure les fichiers CUDA
enable_language(CUDA)

# Ajouter les fichiers source
add_executable(main src/main.cpp src/clarke_wright.cu)
