#ifndef CLARKE_WRIGHT_H
#define CLARKE_WRIGHT_H

#include <vector>
#include <tuple>
#include <thrust/device_vector.h>

using namespace std;

vector<vector<int>> clarke_wright(const thrust::device_vector<float>& locations);

#endif
