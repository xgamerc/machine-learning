#include <stdio.h>  // printf, sprintf
#include <stdlib.h> // srand,rand,RAND_MAX
#include <time.h>   // time
#include <math.h>   // sqrtf

float rand_float() { return (float)rand() / (float)RAND_MAX; }

template <typename T>
void print_array(T* array, size_t size, const char* print_type, char element_prefix = ' ') {
  char format[15] = "\t%c%zu: ";
  sprintf(&format[8], "%s\n", print_type);
  for (size_t i = 0; i < size; ++i) {
    printf(format, element_prefix, i, array[i]);
  }
}

#define PARAMS 4
float data_set[][PARAMS+1] = {
  // {0, 0.05, 0}, {1, 0.05, 2}, {2, 0.05, 4}, {3, 0.05, 6}, {4, 0.05, 8}

  //                   PARAMS
  //    0            1          2            3      = EXPECTED OUTPUT
  { 2.f / 12.f,    0.30,    1.f / 5.f,   3.f / 8.f,     5.0 / 27.0     },
  { 4.f / 12.f,    0.35,    2.f / 5.f,   4.f / 8.f,     8.5 / 27.5     },
  { 6.f / 12.f,    0.40,    2.f / 5.f,   5.f / 8.f,    12.0 / 27.0     },
  { 8.f / 12.f,    0.45,    3.f / 5.f,   6.f / 8.f,    16.0 / 27.0     },
  {10.f / 12.f,    0.50,    4.f / 5.f,   7.f / 8.f,    21.0 / 27.0     },
  {12.f / 12.f,    0.55,    5.f / 5.f,   8.f / 8.f,    27.0 / 27.0     }
};
#define data_size sizeof(data_set) / sizeof(data_set[0])

float cost(float* w, float b, size_t parameters) {
  float result = 0.f;
  for (size_t i = 0; i < data_size; ++i) {
    float y = data_set[i][parameters];
    float output = 0.f;
    for (size_t j = 0; j < parameters; ++j) output += data_set[i][j]*w[j];
    output += b;
    float d = output-y;
    result += d*d;
  }
  result /= (float)data_size;
  return result;
}


void test(float* w, float b, int parameters) {
  printf("\n");
  for (size_t i = 0; i < data_size; ++i) {
    float y = data_set[i][parameters];
    float output = 0.f;
    for (size_t j = 0; j < parameters; ++j) {
      float x = data_set[i][j];
      output += x*w[j];
      printf("\tx%zu: %f\n", j, x);
    }
    output += b;
    printf("  expected: %.7f | output: %.7f\n\n", y, output);
  }
  printf("\n");
}
float grad_norm(float*dw, float db, size_t parameters) {
  float n = db*db;
  for (size_t p = 0; p < parameters; p++) n += dw[p]*dw[p];
  return sqrtf(n);
}
long long seed() {
  /*
    ! with 2600 tries & rate as 1e-2
    ! these seeds give a low cost of 5.13E-4
    ? const long long seed = 1770237910;  // grad_norm is ~0.0003470230
    ? const long long seed = 1770236250;  // grad_norm is ~0.0001033877
  */
  // return time(0);
  return 1770236250;
}
int main(void) {
  
  const long long s = seed();
  srand(s);

  const float rate = 1e-2;
  float  w[PARAMS];
  for (size_t p = 0; p < PARAMS; p++) w[p] = rand_float();
  float b = rand_float();
  float dw[PARAMS];
  float db = 0.f;
  // const long long seed = time(0);
  // srand(42);
  
  printf("----------------------BEFORE LEARNING----------------------\n");
  // printf("Test:"); test(w, b, PARAMS);
  printf("Weights:\n");
  print_array<float>(w, PARAMS, "%f", 'w');
  printf("Cost: %f | bias: %f\n", cost(w, b, PARAMS), b);                      
  printf("grad_norm: %e\n", grad_norm(dw, db, PARAMS));
  // LEARNING / TRAINING
  const size_t tries = 2600;
  for (size_t i = 0; i < tries; i++) {
    db = 0.f;
    for (size_t p = 0; p < PARAMS; p++) dw[p] = 0.f;
    for (size_t i = 0; i < data_size; ++i) {
      float output = 0.f;
      for (size_t p = 0; p < PARAMS; p++) output += data_set[i][p]*w[p];
      output += b; 
      float e = output-data_set[i][PARAMS]; 
      db += 2.f*e;      
      for (size_t p = 0; p < PARAMS; p++) dw[p] += 2.f*e*data_set[i][p];
    }
    for (size_t p = 0; p < PARAMS; p++) {
      dw[p] /= data_size; 
      w[p] -= rate*dw[p]; 
    }
    db /= data_size;      
    b -= rate * db;       
  }
  const float f = 1.033877e-04;
  printf("\n");
  printf("----------------------AFTER LEARNING----------------------\n");
  // printf("Test:"); test(w, b, PARAMS);   
  printf("Weights:\n");
  print_array<float>(w, PARAMS, "%f", 'w');
  printf("Cost: %f | bias: %f\n", cost(w, b, PARAMS), b);                      
  printf("seed: %lld\n", s);
  printf("grad_norm: %e\n", grad_norm(dw, db, PARAMS));
  return 0;
}