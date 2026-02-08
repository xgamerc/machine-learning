#include "utility.h"

#define PARAMS 4
float data_set[][PARAMS+1] = {
  //    0            1          2            3      = EXPECTED OUTPUT
  { 2.f / 12.f,    0.30,    1.f / 5.f,   3.f / 8.f,      5.0 / 27.0     },
  { 4.f / 12.f,    0.35,    2.f / 5.f,   4.f / 8.f,      8.5 / 27.5     },
  { 6.f / 12.f,    0.40,    2.f / 5.f,   5.f / 8.f,     12.0 / 27.0     },
  { 8.f / 12.f,    0.45,    3.f / 5.f,   6.f / 8.f,     16.0 / 27.0     },
  {10.f / 12.f,    0.50,    4.f / 5.f,   7.f / 8.f,     21.0 / 27.0     },
  {12.f / 12.f,    0.55,    5.f / 5.f,   8.f / 8.f,     27.0 / 27.0     }
};
#define data_size sizeof(data_set) / sizeof(data_set[0])


int main(void) {
  const long long s = seed();
  srand(s);
  const float learning_rate = 1e-2;
  float  w[PARAMS];
  for (size_t p = 0; p < PARAMS; p++) w[p] = rand_float();
  float b = rand_float();
  float dw[PARAMS];
  float db = 0.f;
  printf("----------------------BEFORE LEARNING----------------------\n");
  printf("Test:"); test(w, b, PARAMS, (float**)data_set, data_size);
  printf("Weights:\n");
  print_array<float>(w, PARAMS, "%f", 'w');
  printf("Cost: %f | bias: %f\n", cost(w, b, PARAMS, (float**)data_set, data_size), b);                      
  printf("grad_norm: %e\n", grad_norm(dw, db, PARAMS));

  // LEARNING / TRAINING:                           \
  \
  descending a multidimensional quadratic bowl      \
  \
  Batch gradient descent that minimizes mean        \
  squared prediction error by updating parameters   \
  along the negative gradient of the loss function  \
  \
      Parameters are tuned by ERROR FEEDBACK,       \
     Changes are applied evenly and sensitively

  const size_t steps = 2600;
  for (size_t i = 0; i < steps; i++) {
    db = 0.f;
    for (size_t p = 0; p < PARAMS; p++) dw[p] = 0.f;
    for (size_t i = 0; i < data_size; ++i) {
      float output = 0.f;
      for (size_t p = 0; p < PARAMS; p++) output += data_set[i][p]*w[p];
      output += b; 
      float e = output-data_set[i][PARAMS]; 
      db += 2.f*e;      
      for (size_t p = 0; p < PARAMS; p++)
        dw[p] += 2.f*e*data_set[i][p]; // This is sensitivity
    }
    for (size_t p = 0; p < PARAMS; p++) {
      dw[p] /= data_size; 
      w[p] -= learning_rate * dw[p]; 
    }
    db /= data_size;      
    b -= learning_rate * db;       
  }
  // the cost should be negligible after this point 
  // const float f = 1.033877e-04; current grad_norm
  printf("\n");
  printf("----------------------AFTER LEARNING----------------------\n");
  printf("Test:"); test(w, b, PARAMS, (float**)data_set, data_size);   
  printf("Weights:\n");
  print_array<float>(w, PARAMS, "%f", 'w');
  printf("Cost: %f | bias: %f\n", cost(w, b, PARAMS, (float**)data_set, data_size), b);                      
  printf("seed: %lld\n", s);
  printf("grad_norm: %e\n", grad_norm(dw, db, PARAMS));
  return 0;
}