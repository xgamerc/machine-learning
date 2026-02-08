#pragma once
#include <stdlib.h> // srand,rand,RAND_MAX
#include <time.h>   // time
#include "gates.h"  // gates

float rand_float() { return (float)rand() / (float)RAND_MAX; }
long long seed(char randomize = 0) {
  if (randomize) return (time(0));
  /* for ML/src/gradients.cpp
    ! with 2600 tries & rate as 1e-2
    ! these seeds give a low cost of 5.13E-4
    ? const long long seed = 1770237910;  // grad_norm is ~0.0003470230
    ? const long long seed = 1770236250;  // grad_norm is ~0.0001033877
  */
  return (1770236250);
}
#include <stdio.h>  // printf, sprintf
template <typename T>
void print_array(T* array, size_t size, const char* print_type, char element_prefix = ' ') {
  char format[15] = "\t%c%zu: ";
  sprintf(&format[8], "%s\n", print_type);
  for (size_t i = 0; i < size; ++i) {
    printf(format, element_prefix, i, array[i]);
  }
}

#include <math.h>   // sqrtf, expf
float grad_norm(float*dw, float db, size_t parameters) {
  db *= db;
  for (size_t p = 0; p < parameters; p++) db += dw[p]*dw[p];
  return sqrtf(db);
}
float sigmoidf(float x) { return 1.f / (1.f+expf(-x)); }


void test(float* w, float b, size_t parameters, float** data_set, size_t data_size) {
  printf("\n");
  size_t stride = (parameters+1)*sizeof(float);
  for (size_t i = 0; i < data_size; ++i) {
    float * const row = (float*)((char*)data_set+i*stride);
    float y = row[parameters];
    float output = 0.f;
    for (size_t j = 0; j < parameters; ++j) {
      float x = row[j];
      output += x*w[j];
      printf("\tx%zu: %f\n", j, x);
    }
    output += b;
    printf("  expected: %.7f | output: %.7f\n\n", y, output);
  }
  printf("\n");
}
void test(float* w, float b, size_t parameters, float** data_set, size_t data_size, float(*activation_func)(float)) {
  printf("\n");
  size_t stride = (parameters+1)*sizeof(float);
  for (size_t i = 0; i < data_size; ++i) {
    float * const row = (float*)((char*)data_set+i*stride);
    float y = row[parameters];
    float output = 0.f;
    for (size_t j = 0; j < parameters; ++j) {
      float x = row[j];
      output += x*w[j];
      printf("\tx%zu: %f\n", j, x);
    }
    output += b;
    printf("  expected: %.7f | activation_func(output): %.7f\n\n", y, activation_func(output));
  }
  printf("\n");
}

float cost(float* w, float b, size_t parameters, float** data_set, size_t data_size) {
  float result = 0.f;
  size_t stride = (parameters+1)*sizeof(float);
  for (size_t i = 0; i < data_size; ++i) {
    float * const row = (float*)((char*)data_set+i*stride);
    float y = row[parameters];
    float output = 0.f;
    for (size_t j = 0; j < parameters; ++j) {
      output += row[j]*w[j];
    }
    output += b;
    float d = output-y;
    result += d*d;
  }
  return result/(float)data_size;
}
float cost(float* w, float b, size_t parameters, float** data_set, size_t data_size, float(*activation_func)(float)) {
  float result = 0.f;
  size_t stride = (parameters+1)*sizeof(float);
  for (size_t i = 0; i < data_size; ++i) {
    float * const row = (float*)((char*)data_set+i*stride);
    const float y = row[parameters];
    float output = 0.f;
    for (size_t j = 0; j < parameters; ++j) {
      output += row[j]*w[j];
    }
    output += b;
    float d = activation_func(output)-y;
    result += d*d;
  }
  return result/(float)data_size;
}
