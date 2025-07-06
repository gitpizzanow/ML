#include <stdio.h>
#include <stdlib.h>
#include <math.h>


_Float16 sigmoid(_Float16 x){
    return (float) 1/(1+exp(-x));
}

_Float16* getYhat(_Float16* X,float* w,float* b,size_t size,_Float16 threshold){
    _Float16* Yvalues =(_Float16*)calloc(size,sizeof(_Float16));
    for(size_t i=0;i<size;i++){
      float l = (float)X[i]*(*w) + *b;
        Yvalues[i] = (float)sigmoid(l);
    }
    return Yvalues;
}


void metrics(_Float16* X,_Float16* y,float* w,float* b,size_t size,_Float16 threshold){
    float accuracy = 0,loss=0,dw=0,db=0,error=0,lr = 0.05;
    _Float16* yhat = getYhat(X,w,b,size,threshold);
    _Float16 value=0;
    float EPS = 1e-6f;
    for(size_t i=0;i<size;i++){
      value = (float)(yhat[i]>=threshold)?1:0;
        if(value==y[i]) accuracy ++;
        loss+= -(y[i]*logf(yhat[i] + EPS) + (1-y[i])*logf(1-yhat[i] + EPS));
        error= ( yhat[i] - y[i]);
         dw += error * X[i];
         db += error;
    }
    printf("accuracy = %.2f\n",(float)accuracy/size);
    loss/=(float)size;
    printf("loss = %f\n",loss);
    dw/=size;
    db/=size;
    
    *w -= lr * dw;
    *b -= lr * db;
    
printf("w = %f, b = %f\n", *w, *b);
    free(yhat);
}


int main(){
    
    _Float16 X[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    _Float16 y[] = {0, 1, 1, 0,1,1,1,0,1,1};
    float w=4,b=-2.5;
    
    size_t size = sizeof(X)/sizeof(*X);

    for(size_t i=0;i<20;i++){
      printf("epoch %zu : \n",i+1);
      
      metrics(X,y,&w,&b,size,0.5);
      printf("----------------\n");
    }
    
    

    return 0;
}