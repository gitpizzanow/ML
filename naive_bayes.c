#include <stdio.h>
#include <stdlib.h>
/*
    Naive Bayes basic example
    No lissage process for 0 prob
*/
float prior(unsigned char* y,
    unsigned char v,size_t size){
    unsigned char s=0;
    for(size_t i =0;i<size;i++){
        if(y[i]==v) s+=1;
    }
    return (float) s/size;
}

float likelihood(unsigned char* y,
    unsigned char* newsample,
    unsigned char* X,
    unsigned char v,size_t size){

    unsigned char count = 0;
    float p=1;

    for(size_t j=0;j<prior(y,v,size)*size;j++){
    for(size_t i=0;i<size;i++){
        if(y[i]==v){
            if(*(X+j*size + i) ==newsample[j]) count ++;
        }
    }
    //printf("console : %f\n\n", (float)count/(prior(y,v,size)*size));
    p*= (float)count/(prior(y,v,size)*size);
    count =0;
}
    return p;
}

int main(void){
    unsigned char X[][3] = {
        {1,1,1},
        {1,0,1},
        {0,1,0},
        {0,0,1},
        {1,1,0}
    };
    unsigned char y[] = {1,1,0,1,0};
    size_t row = sizeof(X)/sizeof(*X); //5
    size_t col = sizeof(*X)/sizeof(char);//3
    
    unsigned char newsample [] = {1,0,1};// 1 ? 0

    //prior(y,1,row) * 5 // 3 yes : 1
    //prior(y,0,row) * 5 // 2 yes : 0

    printf("they will %s play .\n", (
        likelihood(y,newsample,(char*)X,1,row)*prior(y,1,row)>
        likelihood(y,newsample,(char*)X,0,row)*prior(y,0,row))?"":"not");
    
    return 0;
}