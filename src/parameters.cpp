#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PARAMS 2
float data_set[][3] = {
  {0, -0.05, 0}, 
  {1, -0.05, 2}, 
  {2, -0.05, 4}, 
  {3, -0.05, 6}, 
  {4, -0.05, 8}
};
#define data_size sizeof(data_set) / sizeof(data_set[0])

float cost(float* w, float b, int parameters) {
  float result = 0.0f;
  for (size_t i = 0; i < data_size; ++i) {
    float y = data_set[i][parameters];
    float output = 0.0f;
    for (size_t j = 0; j < parameters; ++j) {
      output += data_set[i][j] * w[j];
    }
    output += b;
    float d = output - y;
    result += d*d;
  }
  result /= (float)data_size;
  return result;
}

void test(float* w, float b, int parameters) {
  for (size_t i = 0; i < data_size; ++i) {
    float y = data_set[i][parameters];
    float output = 0.0f;
    printf("inputs:\n");
    for (size_t j = 0; j < parameters; ++j) {
      float x = data_set[i][j];
      output += x * w[j];
      printf("\tX%zu: %f | W%zu: %f\n", j, x, j, w[j]);
    }
    output += b;
    printf(" expected: %.7f | output: %.7f\n", y, output);
  }
}

float rand_float() { return (float)rand() / (float)RAND_MAX; }

int main(void) {
  srand(time(0));  // srand(42);
  const float eps = 1e-4;
  const float rate = 1e-2;

  float weights[PARAMS] = {};
  for (size_t i = 0; i < data_size; i++) {
    weights[i] = rand_float();  // tells us how strong the connection is for each input
  }
  float b = rand_float();
  
  printf("--------------------------BEFORE LEARNING--------------------------\n");
  test(weights, b, PARAMS);

  // LEARNING
  float dweights[PARAMS] = {};
  for (size_t i = 0; i < 1500; i++) {
    float c = cost(weights, b, PARAMS);
    for (size_t p = 0; p < PARAMS; p++) {
      weights[p] += eps;
      dweights[p] = (cost(weights, b, PARAMS) - c)/eps;
      weights[p] -= eps;
    }
    float db = (cost(weights, b+eps, PARAMS) - c)/eps;

    for (size_t p = 0; p < PARAMS; p++) {
      weights[p] -= rate*dweights[p];
    }
    b -= rate*db;
  }
  
  
  printf("\n");
  printf("--------------------------AFTER LEARNING--------------------------\n");
  test(weights, b, PARAMS);   
  printf("\nCost: %f | bias: %f\n", cost(weights, b, PARAMS), b);                      
  return 0;
}