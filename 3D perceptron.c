#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void display_matrix(float* X, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("%.0f ", X[i * cols + j]);
    }
    printf("\n");
  }
}

void train(float* X, float* y, float* yhat, float* weights, size_t rows,
           size_t cols, size_t epoch) {
  float lr = 0.1;

  for (size_t e = 0; e < epoch; e++) {
    size_t p = 0;

    printf("epoch n°%zu : \n", e);
    for (size_t j = 0; j < rows; j++) {
      float sum = 0, loss = 0;

      for (size_t i = 0; i < cols; i++) {
        sum += (X[j * cols + i] * weights[i]);
      }
      yhat[j] = (sum > 0) ? 1 : 0;
      loss = y[j] - yhat[j];

      for (size_t i = 0; i < cols; i++)
        weights[i] += (lr * loss * X[j * cols + i]);
      if (yhat[j] == y[j]) p++;
    }

    for (size_t i = 0; i < rows; i++) printf("%.2f ", yhat[i]);
    printf("\n");
    printf("------------\n");
    if (p == rows) return;
  }
}
int main(void) {
  unsigned int seed = 50;
  srand(seed);

  size_t rows = 4, cols = 3;
  float X[][3] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 0, 0}};
  float y[] = {0, 1, 1, 1};
  float yhat[rows];
  float weights[] = {0.2, 0.3, 0.04};
  // for (size_t i = 0; i < cols; i++)
  //   weights[i] = (float)  rand() / (RAND_MAX + 1.0f);

  /*
  data :
             xor
  0.00 0.00   0
  1.00 0.00   1
  0.00 1.00   1
  1.00 1.00   0

  y = [0 1 1 1]
  w= [0.5  0.2]

  */

  train((float*)X, y, yhat, weights, rows, cols, 10);
}
