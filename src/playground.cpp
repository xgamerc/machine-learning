
#include "gates.h"
#include "utility.h"

#define PARAMS 2
#define WEIGHTS PARAMS
#define entries_amount sizeof(OR_table) / sizeof(OR_table[0])

void print_xor(Xor* m) {
  printf("   m.or_w1: %f\n", m->or_w1);
  printf("   m.or_w2: %f\n", m->or_w2);
  printf("    m.or_b: %f\n", m->or_b);
  printf(" m.nand_w1: %f\n", m->nand_w1);
  printf(" m.nand_w2: %f\n", m->nand_w2);
  printf("  m.nand_b: %f\n", m->nand_b);
  printf("  m.and_w1: %f\n", m->and_w1);
  printf("  m.and_w2: %f\n", m->and_w2);
  printf("   m.and_b: %f\n", m->and_b);
}
float forward(Xor m, float x, float y) {
  float a = sigmoidf(m.or_w1*x + m.or_w2*y + m.or_b);         //  first neuron first layer
  float b = sigmoidf(m.nand_w1*x + m.nand_w2*y + m.nand_b);   // second neuron first layer
  return sigmoidf(m.and_w1*a + m.and_w2*b + m.and_b);         // first neuron second layer
}
float cost(Xor m, float data_set[][3], size_t data_size) {
  float result = 0.f;
  for (size_t i = 0; i < data_size; ++i) {
    float output = forward(m, data_set[i][0], data_set[i][1]);
    float d = output-data_set[i][2];
    result += d*d;
  }
  return result/(float)data_size;
}
void test(Xor m, float data_set[][3], size_t data_size) {
  printf("\n");
  for (size_t i = 0; i < data_size; ++i) {
    printf("\tx0: %f\n", data_set[i][0]);
    printf("\tx1: %f\n", data_set[i][1]);
    float output = forward(m, data_set[i][0], data_set[i][1]);
    float y = data_set[i][2];
    printf("  expected: %.7f | output: %.7f\n\n", y, output);
  }
  printf("\n");
}
Xor finite_differences(Xor* m, float data_set[][3], size_t data_size, float eps) {
  Xor res;
  float c = cost(*m, data_set, data_size);
  float temp = .0f;

  temp = m->or_w1;
  m->or_w1 += eps;
  res.or_w1 = (cost(*m, data_set, data_size) - c)/eps;
  m->or_w1 = temp;

  temp = m->or_w2;
  m->or_w2 += eps;
  res.or_w2 = (cost(*m, data_set, data_size) - c)/eps;
  m->or_w2 = temp;
  
  temp = m->or_b;
  m->or_b += eps;
  res.or_b = (cost(*m, data_set, data_size) - c)/eps;
  m->or_b = temp;
  
  temp = m->nand_w1;
  m->nand_w1 += eps;
  res.nand_w1 = (cost(*m, data_set, data_size) - c)/eps;
  m->nand_w1 = temp;
  
  temp = m->nand_w2;
  m->nand_w2 += eps;
  res.nand_w2 = (cost(*m, data_set, data_size) - c)/eps;
  m->nand_w2 = temp;
  
  temp = m->nand_b;
  m->nand_b += eps;
  res.nand_b = (cost(*m, data_set, data_size) - c)/eps;
  m->nand_b = temp;
  
  temp = m->and_w1;
  m->and_w1 += eps;
  res.and_w1 = (cost(*m, data_set, data_size) - c)/eps;
  m->and_w1 = temp;
  
  temp = m->and_w2;
  m->and_w2 += eps;
  res.and_w2 = (cost(*m, data_set, data_size) - c)/eps;
  m->and_w2 = temp;
  
  temp = m->and_b;
  m->and_b += eps;
  res.and_b = (cost(*m, data_set, data_size) - c)/eps;
  m->and_b = temp;

  return res;
}
void learn(Xor* model, Xor diff, float rate) {
  model->or_w1 -= rate*diff.or_w1;
  model->or_w2 -= rate*diff.or_w2;
  model->or_b -= rate*diff.or_b;
  model->nand_w1 -= rate*diff.nand_w1;
  model->nand_w2 -= rate*diff.nand_w2;
  model->nand_b -= rate*diff.nand_b;
  model->and_w1 -= rate*diff.and_w1;
  model->and_w2 -= rate*diff.and_w2;
  model->and_b -= rate*diff.and_b;
}

int main(void) {
  Xor model = rand_xor();
  const long long s = seed('Y');
  srand(s);
  const float eps = 1e-3;
  const float learning_rate = 1e-2;
  printf("----------------------BEFORE LEARNING----------------------\n");
  printf("\n");
  print_xor(&model);
  // data_point* test_table = OR_table;
  data_point* test_table = NOR_table;
  // data_point* test_table = NAND_table;
  // data_point* test_table = AND_table;
  // data_point* test_table = XOR_table;
  float c = cost(model, test_table, entries_amount);
  printf("\n");
  test(model, test_table, entries_amount);
  printf("starting cost(w, b, p, d, s): %f\n", c);
  for (size_t t = 0; t < 1'000'000; t++) {
    // wiggle around 9 parameters and reduce the cost in the most efficient way
    learn(
      &model, 
      finite_differences(&model, test_table, entries_amount, eps), 
      learning_rate
    );
  }
  printf("----------------------AFTER LEARNING----------------------\n");
  printf("\n");
  print_xor(&model);
  float nc = cost(model, test_table, entries_amount);
  printf("\n");
  test(model, test_table, entries_amount);
  printf("   final cost(w, b, p, d, s): %f\n", nc);
  
  printf("\n\n");
  printf("      model output:\n");
  for (int i = 0; i < 2; i++) {
    for (int p = 0; p < 2; p++) {
      printf("  %d, %d   = expected: %f | model: %f\n",i, p, test_table[i*2+p][2], forward(model, i, p));
    }
  }
  printf("  OR neuron output:\n");
  for (int i = 0; i < 2; i++) {
    for (int p = 0; p < 2; p++) {
      printf("  %d | %d  = expected: %f | neuron: %f\n",  i, p, test_table[i*2+p][2], sigmoidf(model.or_w1*i + model.or_w2*p + model.or_b));
    }
  }
  printf("NAND neuron output:\n");
  for (int i = 0; i < 2; i++) {
    for (int p = 0; p < 2; p++) {
      printf("~(%d & %d) = expected: %f | neuron: %f\n",  i, p, test_table[i*2+p][2], sigmoidf(model.nand_w1*i + model.nand_w2*p + model.nand_b));
    }
  }
  printf(" AND neuron output:\n");
  for (int i = 0; i < 2; i++) {
    for (int p = 0; p < 2; p++) {
      printf("  %d & %d  = expected: %f | neuron: %f\n",  i, p, test_table[i*2+p][2], sigmoidf(model.and_w1*i + model.and_w2*p + model.and_b));
    }
  }
  printf("\nseed: %lld\n",s);
  return 0;
}