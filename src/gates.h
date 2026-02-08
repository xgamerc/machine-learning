#pragma once
typedef float data_point[3];

/*
XOR GATE (x | y) & ~(x & y)
  x          x 
  =|         =|
  OR=======NAND
  =|   ||    =|
  y    ||    y 
      AND      
*/

data_point XOR_table[] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};
data_point OR_table[] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1}
};
data_point NOR_table[] = {
  {0, 0, 1},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 0}
};
data_point AND_table[] = {
  {0, 0, 0},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1}
};
data_point NAND_table[] = {
  {0, 0, 1},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};

typedef struct {
  float or_w1;
  float or_w2;
  float or_b;
  float nand_w1;
  float nand_w2;
  float nand_b;
  float and_w1;
  float and_w2;
  float and_b;
} Xor;

float rand_float();
Xor rand_xor() {
  return Xor{
    .or_w1 = rand_float(),
    .or_w2 = rand_float(),
     .or_b = rand_float(),
  .nand_w1 = rand_float(),
  .nand_w2 = rand_float(),
   .nand_b = rand_float(),
   .and_w1 = rand_float(),
   .and_w2 = rand_float(),
    .and_b = rand_float()
  };
}