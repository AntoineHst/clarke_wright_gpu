cmake_minimum_required(VERSION 3.10)
project(VRP)

# Spécifiez le chemin vers le toolkit CUDA
find_package(CUDA REQUIRED)

# Inclure les répertoires CUDA et Thrust
include_directories(${CUDA_INCLUDE_DIRS})

# Ajouter les fichiers sources
set(SOURCES
    src/main.cpp
    src/clarke_wright.cu
)

# Créer l'exécutable
cuda_add_executable(main ${SOURCES})

# Lier CUDA avec votre exécutable
target_link_libraries(main ${CUDA_LIBRARIES})
