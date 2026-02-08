
#include "utility.h"

#define PARAMS 2
#define WEIGHTS PARAMS


// OR GATE                       
float data_set[][PARAMS+1] = {   
  {0, 0, 0},                     
  {1, 0, 1},                     
  {0, 1, 1},                     
  {1, 1, 1}                      
};

// AND GATE                      
// float data_set[][PARAMS+1] = {
  // {0, 0, 0},                  
  // {1, 0, 0},                  
  // {0, 1, 0},                  
  // {1, 1, 1}                   
// };

// NAND GATE                     
// float data_set[][PARAMS+1] = {
  // {0, 0, 1},                  
  // {1, 0, 1},                  
  // {0, 1, 1},                  
  // {1, 1, 0}                   
// };

#define data_size sizeof(data_set) / sizeof(data_set[0])

int main(void) {
  const long long s = seed(1);
  srand(s);

  const float eps = 1e-3;
  const float learning_rate = 1e-2;
  
  float  weights[PARAMS];
  for (size_t p = 0; p < PARAMS; p++) weights[p] = rand_float();
  // float bias = rand_float();
  float bias = 0.25f; // without a bias, the model can only use the weights
  // but with a bias, the entire thing can shift as a whole, overall
  // bias adds a degree of freedom
  
  printf("----------------------BEFORE LEARNING----------------------\n");
  printf("\n");
  print_array<float>(weights, PARAMS, "%f", 'w');
  test(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf);
  float c = cost(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf);
  printf("starting cost(w, b, p, d, s): %f | bias: %f\n", c, bias);
  
  
  // printf("COST PRINT START\n");                             //////
  float dweights[WEIGHTS] = {};
  for (size_t t = 0; t < 1'892'000; t++) {
    // current cost
    c = cost(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf); 
    // printf("COST: %f\n", c);                             //////
    for (size_t w = 0; w < WEIGHTS; w++) { 
      weights[w] += eps; // this practice is just for learning purposes, to make the process clear
      float c2 = cost(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf);
      weights[w] -= eps; // this practice is just for learning purposes, to make the process clear
      dweights[w] = (c2 - c)/eps;
    }
    bias += eps; // this practice is just for learning purposes, to make the process clear
    float cb = cost(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf);
    bias -= eps; // this practice is just for learning purposes, to make the process clear
    float db = (cb - c)/eps;
    bias -= learning_rate*db;
    for (size_t w = 0; w < WEIGHTS; w++) {
      weights[w] -= learning_rate*dweights[w];
    }
  }
  // printf("COST PRINT END\n");                             //////

  printf("----------------------AFTER LEARNING----------------------\n");
  printf("\n");
  print_array<float>(weights, PARAMS, "%f", 'w');
  test(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf);
  float nc = cost(weights, bias, PARAMS, (float**)data_set, data_size, sigmoidf);
  printf("   final cost(w, b, p, d, s): %f | bias: %f\n", nc, bias);
  printf("seed: %lld\n",s);
  return 0;
}