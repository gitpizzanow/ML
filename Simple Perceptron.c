#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/*
Simple Perceptron
*/

void display(float* T , size_t size){
  printf("weights =");
    printf(" [ ");
     for(size_t i=0;i<size;i++) printf("%.3f ", T[i]);
      printf("]\n");
}
/*activation function
step function
    yi = 1 if >0 else 0

*/

float sum(float* T , size_t size){
    float s=0;
    for(size_t i=0;i<size;i++) s+= T[i];
}

float activation(float a){
    return (a>0)?1:0;
}


void train(float* X ,float* yhat, float* target,size_t row,size_t col,float* w,size_t epoch ){
  
  for(size_t i=0;i<epoch;i++){
    unsigned char correct = 0;
     printf("epoch %zu :\n",i);
    float alpha = .1;
    float p =0;

    for(size_t j=0;j<row;j++){
       
        float sum=0;
        for(size_t i=0;i<col;i++) sum +=  X[j*col + i] * w[i];
        
        yhat[j] = sum;
        float prediction = activation(yhat[j]);
        p+=prediction;
       
        float error = target[j]-prediction;
        
         for(size_t i=0;i<col;i++) {
          w[i]+=alpha * error * X[j*col + i];
          
         }
         if(target[j] == prediction) correct++;
    }
    display(w,row);
  
     if(correct ==row) {
       display(w,row);
       return;
     }
  }
 
}


int main(void){
  float X[][3]= {
    {1,0,2},
    {0,1,1},
    {2,0,1}
  };
  float target[] = {1,0,1};
  float w[3];
  for (int i = 0; i < 3; i++)
        w[i] = (float)rand() / RAND_MAX;
    
  float* yhat = (float*)calloc(3,sizeof(float));
  
  size_t row = sizeof(X)/sizeof(*X);
  size_t col = sizeof(*X)/sizeof(float);
  

  train((float*)X ,yhat, target, row,col,w,10 );
}
