#include "clarke_wright.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdint> // Inclure cstdint pour les types de taille fixe

using namespace std;
using namespace thrust;

// Fonction pour calculer la distance euclidienne entre deux points
__device__ float euclidean_distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Noyau pour calculer les distances entre tous les points
__global__ void compute_distances(const float* locations, float* distances, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        float x1 = locations[i * 2];
        float y1 = locations[i * 2 + 1];
        float x2 = locations[j * 2];
        float y2 = locations[j * 2 + 1];
        distances[i * n + j] = euclidean_distance(x1, y1, x2, y2);
    }
}

// Fonction Clarke-Wright pour calculer les routes
vector<vector<int64_t>> clarke_wright(const thrust::device_vector<float>& d_locations) {
    int64_t n = d_locations.size() / 2 - 1;
    thrust::device_vector<float> d_distances(n * n);

    // Définir les dimensions du noyau
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Lancer le noyau pour remplir les distances
    compute_distances<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(d_locations.data()), thrust::raw_pointer_cast(d_distances.data()), n);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_distances = d_distances;
    vector<thrust::tuple<float, int64_t, int64_t>> savings;

    // Calculer les économies et trier
    for (int64_t i = 1; i <= n; ++i) {
        for (int64_t j = i + 1; j <= n; ++j) {
            float saving = h_distances[i * (n + 1) + 0] + h_distances[j * (n + 1) + 0] - h_distances[i * (n + 1) + j];
            savings.push_back(thrust::make_tuple(saving, i, j));
        }
    }

    // Trier les économies en ordre décroissant
    thrust::sort(savings.begin(), savings.end(), thrust::greater<thrust::tuple<float, int64_t, int64_t>>());

    vector<vector<int64_t>> routes(n);
    for (int64_t i = 0; i < n; ++i) {
        routes[i].push_back(i + 1);
    }

    // Construire les routes
    for (const auto& saving : savings) {
        float s;
        int64_t i, j;
        thrust::tie(s, i, j) = saving;
        if (!routes[i - 1].empty() && !routes[j - 1].empty() && routes[i - 1] != routes[j - 1]) {
            routes[i - 1].insert(routes[i - 1].end(), routes[j - 1].begin(), routes[j - 1].end());
            routes[j - 1].clear();
        }
    }

    vector<vector<int64_t>> final_routes;
    for (const auto& route : routes) {
        if (!route.empty()) {
            final_routes.push_back(route);
        }
    }

    return final_routes;
}
