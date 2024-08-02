#ifndef CLARKE_WRIGHT_H
#define CLARKE_WRIGHT_H

#include <vector>
#include <thrust/device_vector.h>

// Déclaration de la fonction euclidean_distance pour éviter la redéfinition
__device__ float euclidean_distance(float x1, float y1, float x2, float y2);

// Déclaration de la fonction compute_distances pour éviter la redéfinition
__global__ void compute_distances(const float* locations, float* distances, int64_t n);

// Déclaration de la fonction clarke_wright
std::vector<std::vector<int64_t>> clarke_wright(const thrust::device_vector<float>& d_locations);

#endif // CLARKE_WRIGHT_H
