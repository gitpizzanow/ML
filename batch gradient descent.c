#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define EPSILON 1e-7f

/*
  batch gradient descent
*/

float cross(float y, float Yhat) {
  Yhat = fmaxf(Yhat, EPSILON);
  Yhat = fminf(Yhat, 1.0f - EPSILON);
  return -(y * logf(Yhat) + (1 - y) * logf(1.0f - Yhat));
}

float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

float derivative(float z) {
  return z * (1 - z);  // sigmoid(z) * 1-sigmoid(z)
}

void MLP(float* X, float* y, float* w, float* b, float* H, size_t rows,
         size_t cols, size_t len, size_t epochs) {
  float lr = 0.5f;
  size_t p = 0;
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    if (p == 3) return;
    p = 0;
    float loss = 0.0f;

    float dv[3] = {0};  // output layer: v1, v2, b3
    float dw[6] = {0};  // input to hidden w and b

    printf("epoch %zu : \n", epoch);
    printf("original Y : ");
    for (size_t i = 0; i < rows; i++) printf("%d ", (int)y[i]);
    printf("\n");

    printf("original ŷ : ");
    for (size_t k = 0; k < rows; k++) {
      float Yhat = 0;

      // Forward: input to hidden
      for (size_t j = 0; j < cols; j++) {
        float z = 0;
        for (size_t i = 0; i < cols; i++) {
          z += X[k * cols + i] * w[i * cols + j];
        }
        H[j] = sigmoid(z + b[j]);
      }

      // Forward: hidden to output
      for (size_t j = 0; j < cols; j++) {
        Yhat += H[j] * w[len - 2 + j];  // v1, v2
      }
      Yhat += b[cols];  // b3

      float pred = sigmoid(Yhat);
      float error = pred - y[k];
      loss += cross(y[k], pred);

      for (size_t j = 0; j < cols; j++) {
        dv[j] += error * H[j];  // dL/dvj
      }
      dv[2] += error;  // b3

      for (size_t j = 0; j < cols; j++) {
        float dH = error * w[len - 2 + j] * derivative(H[j]);
        for (size_t i = 0; i < cols; i++) {
          dw[i * cols + j] += X[k * cols + i] * dH;
        }
        dw[4 + j] += dH;
      }
      float realP = (pred >= 0.5f) ? 1 : 0;
      if (y[k] == (int)realP) p++;
      printf("%d ", (pred >= 0.5f) ? 1 : 0);
    }

    printf("\n-------------\n");

    for (size_t j = 0; j < cols; j++) {
      w[len - 2 + j] -= lr * dv[j];  // v1, v2
    }
    b[cols] -= lr * dv[2];  // b3

    for (size_t i = 0; i < cols; i++) {
      for (size_t j = 0; j < cols; j++) {
        w[i * cols + j] -= lr * dw[i * cols + j];
      }
    }
    for (size_t j = 0; j < cols; j++) {
      b[j] -= lr * dw[4 + j];
    }
  }
}

int main() {
  float X[][2] = {
      {0, 0},  // 0
      {1, 1},  // 0
      {1, 0}   // 1
  };

  float y[] = {0, 0, 1};

  // Weights: [w1, w2, w3, w4, v1, v2]
  float weights[] = {0.05f, -0.1f, 0.08f, 0.02f, 0.1f, -0.05f};

  float bayes[] = {0.0f, 0.0f, 0.0f};

  float H[2];  // Hidden layer output

  size_t rows = sizeof(X) / sizeof(*X);
  size_t cols = sizeof(*X) / sizeof(float);

  MLP((float*)X, y, weights, bayes, H, rows, cols,
      sizeof(weights) / sizeof(float), 500);

  return 0;
}
