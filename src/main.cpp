#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib> // Pour std::rand()
#include <ctime>   // Pour std::time()
#include "clarke_wright.h"

using namespace std;

int main() {
    // Initialiser le générateur de nombres aléatoires
    std::srand(std::time(0));

    // Crée un vecteur h_locations
    thrust::host_vector<float> h_locations;

    // Ajouter le dépôt
    h_locations.push_back(0.0); // Dépôt
    h_locations.push_back(0.0);

    // Ajouter 700 clients avec des coordonnées aléatoires
    const int num_clients = 700;
    for (int i = 0; i < num_clients; ++i) {
        float x = static_cast<float>(std::rand() % 100); // Coordonnée x aléatoire
        float y = static_cast<float>(std::rand() % 100); // Coordonnée y aléatoire
        h_locations.push_back(x); // Client x
        h_locations.push_back(y); // Client y
    }

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
