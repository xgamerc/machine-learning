
// https://www.youtube.com/watch?v=PGSba51aRYU
#include  <stdio.h>
#include <stdlib.h>
#include   <time.h>

float data_set[][2] = {
  {0, 0},
  {1, 5},
  {2, 10},
  {3, 15},
  {4, 20}
};

#define data_size sizeof(data_set)/sizeof(data_set[0])

float cost(float w, float b) {
  float result = 0.f;
  for (size_t i = 0; i < data_size; i++) {
    float x = data_set[i][0];
    float y = data_set[i][1];
    float output = x*w + b;
    float d = output - y;
    result += d*d;
  }
  result /= (float)data_size;
  return result;
}

void test(float w, float b) {
  for (size_t i = 0; i < data_size; i++) {
    float x = data_set[i][0];
    float y = data_set[i][1];
    float output = x*w + b;
    printf("input: %.7f | expected: %.7f | output: %.7f\n", x, y, output);
  }
}

float rand_float() {
  return (float)rand()/(float)RAND_MAX;
}

int main(void) {
  srand(time(0)); // srand(42);
  const float eps = 1e-4;
  const float rate = 1e-2;
  float w = rand_float(); // tells us how strong the connection is
  float b = rand_float(); // bias is an offset, shifts the activation function to better fit data, Enables Non-Zero Output, acts as threshold for activation functions
  printf("weight: %.7f | bias: %.7f | cost = %.7f (initial)\n", w, b, cost(w, b));
  printf("-----------------------------BEFORE LEARNING---------------------------\n");
  test(w, b);
  
  // Learning and adjusting for the proper the weight and bias
  for (size_t i = 0; i < 1500; i++) {
    float c = cost(w, b);
    float dwcost = (cost(w + eps, b) - c)/eps;
    float dbcost = (cost(w, b + eps) - c)/eps;
    w -= rate*dwcost;
    b -= rate*dbcost;
  }
  printf("\nweight: %.7f | bias: %.7f | cost = %.7f (updated)\n", w, b, cost(w, b));
  printf("-----------------------------AFTER LEARNING----------------------------\n");
  test(w, b);
  
  return 0;
}