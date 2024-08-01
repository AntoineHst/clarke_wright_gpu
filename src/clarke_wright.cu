#include "clarke_wright.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <cmath>

__device__ float euclidean_distance(float x1, float y1, float x2, float y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

vector<vector<int>> clarke_wright(const thrust::device_vector<float>& locations) {
    int n = locations.size() / 2 - 1;
    thrust::device_vector<float> distances((n+1)*(n+1));

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distances[i*(n+1) + j] = euclidean_distance(locations[i*2], locations[i*2+1], locations[j*2], locations[j*2+1]);
        }
    }

    vector<tuple<float, int, int>> savings;
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            float saving = distances[i] + distances[j] - distances[i*(n+1) + j];
            savings.push_back(make_tuple(saving, i, j));
        }
    }

    sort(savings.rbegin(), savings.rend());

    vector<vector<int>> routes(n);
    for (int i = 0; i < n; ++i) {
        routes[i].push_back(i + 1);
    }

    for (auto& saving : savings) {
        float s;
        int i, j;
        tie(s, i, j) = saving;
        if (!routes[i-1].empty() && !routes[j-1].empty() && routes[i-1] != routes[j-1]) {
            routes[i-1].insert(routes[i-1].end(), routes[j-1].begin(), routes[j-1].end());
            routes[j-1].clear();
        }
    }

    vector<vector<int>> final_routes;
    for (auto& route : routes) {
        if (!route.empty()) {
            final_routes.push_back(route);
        }
    }

    return final_routes;
}
