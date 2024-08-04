#include "clarke_wright.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <vector>
#include <cstdint> // Inclure cstdint pour les types de taille fixe

using namespace std;
using namespace thrust;

struct Challenge {
    uint32_t seed;
    struct Difficulty {
        size_t num_nodes;
        uint32_t better_than_baseline;
    } difficulty;
    vector<int32_t> demands;
    vector<vector<int32_t>> distance_matrix;
    int32_t max_total_distance;
    int32_t max_capacity;
};

struct Solution {
    vector<vector<size_t>> routes;
};

Solution clarke_wright(const Challenge& challenge) {
    size_t n = challenge.difficulty.num_nodes;

    // Extraire la matrice de distances
    vector<int32_t> h_distances_flat;
    for (const auto& row : challenge.distance_matrix) {
        h_distances_flat.insert(h_distances_flat.end(), row.begin(), row.end());
    }

    // Vector pour stocker les économies
    vector<thrust::tuple<int32_t, size_t, size_t>> savings;

    // Calculer les économies et trier
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = i + 1; j <= n; ++j) {
            int32_t saving = h_distances_flat[i * (n + 1) + 0] + h_distances_flat[j * (n + 1) + 0] - h_distances_flat[i * (n + 1) + j];
            savings.push_back(thrust::make_tuple(saving, i, j));
        }
    }

    // Trier les économies en ordre décroissant
    thrust::sort(savings.begin(), savings.end(), thrust::greater<thrust::tuple<int32_t, size_t, size_t>>());

    vector<vector<size_t>> routes(n);
    for (size_t i = 0; i < n; ++i) {
        routes[i].push_back(i + 1);
    }

    // Construire les routes
    for (const auto& saving : savings) {
        int32_t s;
        size_t i, j;
        thrust::tie(s, i, j) = saving;
        if (!routes[i - 1].empty() && !routes[j - 1].empty() && routes[i - 1] != routes[j - 1]) {
            routes[i - 1].insert(routes[i - 1].end(), routes[j - 1].begin(), routes[j - 1].end());
            routes[j - 1].clear();
        }
    }

    vector<vector<size_t>> final_routes;
    for (const auto& route : routes) {
        if (!route.empty()) {
            final_routes.push_back(route);
        }
    }

    Solution solution;
    solution.routes = final_routes;
    return solution;
}