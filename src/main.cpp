#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include "clarke_wright.h"

using namespace std;

int main() {
    thrust::host_vector<float> h_locations = {
        0.0, 0.0,  // Dépôt
        1.0, 3.0,  // Client 1
        4.0, 4.0,  // Client 2
        5.0, 1.0   // Client 3
    };

    thrust::device_vector<float> d_locations = h_locations;

    vector<vector<int>> routes = clarke_wright(d_locations);

    for (const auto& route : routes) {
        cout << "Route: [0] -> ";
        for (int loc : route) {
            cout << "[" << loc << "] -> ";
        }
        cout << "[0]" << endl;
    }

    return 0;
}
