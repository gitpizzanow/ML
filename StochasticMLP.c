#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ROWS 3
#define COLS 2
#define N 4

float sigmoid(float a) { return 1.0f / (1 + exp(-a)); }

int step(float a) { return (a >= 0.5) ? 1 : 0; }

void backward(float* X, float* weights, float* bayes, float* dv, float* db,
              float* H,float* pred, const float* y, int k,
              float lr) {
  float dh[N];
  float error = pred[k] - y[k];

  // use dv to compute dh
  for (size_t i = 0; i < N; i++) dh[i] = error * dv[i] * H[i] * (1.0f - H[i]);

  // update dv and db
  for (size_t i = 0; i < N; i++) dv[i] -= lr * error * H[i];

  (*db) -= lr * error;

  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < COLS; i++) {
      weights[j * COLS + i] -= lr * dh[j] * X[k * COLS + i];
    }
    bayes[j] -= lr * dh[j];
  }
}

void forward(float* X, float* weights, float* bayes, float* dv, float* db,
             float* H, float* pred, const float* y, int k) {
  for (size_t j = 0; j < N; j++) {
    float Z = 0;
    for (size_t i = 0; i < COLS; i++)
      Z += X[k * COLS + i] * weights[j * COLS + i];

    H[j] = sigmoid(Z + bayes[j]);
    // printf("Z%zu = %.2f \n",j ,H[j]);
  }

  float yhat = 0, predict = 0;
  for (size_t j = 0; j < N; j++) yhat += (H[j] * dv[j]);

  predict = sigmoid(yhat + *db);
  pred[k] = predict;
  // printf("Y%d = %.0f , pred = %.2f , ŷ=%d\n", k, y[k], predict,
  // step(predict));
  printf("Y%d = %.0f , ŷ=%d\n", k, y[k], step(predict));
}

int main() {
  float X[][2] = {
      {1, 0},  // 1
      {0, 0},  // 0
      {1, 1}   //  0
  };
  float y[] = {1, 0, 0};
  float H[10]; //one hidden layer
  float weights[ROWS * COLS];
  // w11, w21 -> h1 | w12, w22 -> h2 | w13, w33 -> h3
  float bayes[N];
  float dv[N];
  float db;

  srand(time(NULL));
  for (size_t i = 0; i < N * COLS; i++)
    weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  for (size_t i = 0; i < N; i++) {
    bayes[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    dv[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  }
  db = ((float)rand() / RAND_MAX - 0.5f) * 2;
  float pred[3];

  float lr = 0.1;
  size_t epochs = 700;

  // training

  for (size_t e = 0; e < epochs; e++) {
    int p = 0;
    printf("epoch°%zu :\n", e);
    for (size_t i = 0; i < ROWS; i++) {
      forward((float*)X, weights, bayes, dv, &db, H, pred, y, i);

      if ((int)y[i] == step(pred[i])) p++;
      backward((float*)X, weights, bayes, dv, &db, H, pred, y, i, lr);
    }

    if (p == ROWS) {
      printf("Stopping at epoch %zu.\n", e);
      break;
    }
  }
}
