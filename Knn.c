#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    _Float16 x,y;
}Point;

void display(_Float16* T,size_t size){
    for(char i =0;i<size;i++) printf("%.2f ",(float)*(T+i));
}

_Float16 distance(_Float16 xa ,_Float16 xb,  _Float16 ya ,_Float16 yb){
    _Float16 d1 = (xa - xb)*(xa - xb);
    _Float16 d2 = (ya - yb)*(ya - yb);
    return (_Float16) sqrt(d1 + d2);
}

_Float16 getMin(_Float16* T,_Float16* y,
    size_t size,unsigned char K,
    unsigned short* label1, unsigned short* label2){
    if(K<1) return 0;
    unsigned char min_idx = 0;
    for(size_t i =0;i<size;i++){
        if(*(T+i)< *(T+min_idx))min_idx =i;
    }
    (y[min_idx]==1)?(*label1)++ : (*label2)++;
    _Float16 current_min = *(T + min_idx);  
    *(T+min_idx) = (_Float16)1e5;
    K--;
    getMin(T,y,size,K,label1,label2);
    return current_min;
}


int main(int argc , char* argv[]){
    unsigned k =3;
    Point points[] = {{6,8},{17,20},{5,7},{12,16},{16,18}};
    _Float16 y[] = {1,0,1,0,0};
    Point newPoint = {10,10};
    size_t size = sizeof(y)/sizeof(*y);
     _Float16 distances[size];
    for(char i =0;i<size;i++){
        distances[i] = (float)distance(points[i].x,newPoint.x,points[i].y,newPoint.y);
    };
    unsigned short label1=0,label2=0;
    getMin(distances,y,size,k,&label1,&label2);
    printf("the new point is likely an %s \n",(label1>label2)?"apple":"banana");
    
}