#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "clarke_wright.h"

using namespace std;

int main() {
    // Crée un vecteur h_locations et ajoute les éléments un par un
    thrust::host_vector<float> h_locations;
    h_locations.push_back(0.0); // Dépôt
    h_locations.push_back(0.0);
    h_locations.push_back(1.0); // Client 1
    h_locations.push_back(3.0);
    h_locations.push_back(4.0); // Client 2
    h_locations.push_back(4.0);
    h_locations.push_back(5.0); // Client 3
    h_locations.push_back(1.0);

    // Copie les données du vecteur h_locations vers d_locations
    thrust::device_vector<float> d_locations = h_locations;

    // Appel à la fonction clarke_wright avec le vecteur d_locations
    vector<vector<int>> routes = clarke_wright(d_locations);

    // Affichage des résultats
    for (const auto& route : routes) {
        cout << "Route: [0] -> ";
        for (int loc : route) {
            cout << "[" << loc << "] -> ";
        }
        cout << "[0]" << endl;
    }

    return 0;
}
