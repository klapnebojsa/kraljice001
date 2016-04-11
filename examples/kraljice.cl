__kernel void kraljice_brojac(__global int* data,
                              __global int* brpolja,
                              __local float* partial_sums,
                              __global float* output) {   
    int brP=6;
    int k,ind;
    int x[6];
    //char *resenje[brpolja];
    //int resenje;
    //float sum[get_local_id(0)]=0.0;
    

    
    int lid = get_local_id(0);
    //printf("%u\n", get_local_id(0));    
    partial_sums[lid]=0.0;
    
    x[1]=0;
    k=1;
    while (k > 0) {
        x[k]++;
        if (x[k]<= brP){
            if (k == brP) {
                //resenje=0;
                for (ind=1;ind<=brP; ind++) {partial_sums[lid] +=(float)x[ind];}
                                                //{sprintf(resenje, "%d", x[ind]);}
            }
            else {k++; x[k]=0;}
         }
         else {k--;}
    }
    //sum =(float)resenje;
    if(lid == 0) {
        output[get_group_id(0)] = partial_sums[lid];
    }
}






__kernel void kraljice_brojac1(__global float* data,
                               //__local float* partial_sums,
                               __global float* output) {   
    int brPolja=4;
    int n=4;
    int k,ind;
    int x[4];
    //char *resenje[brPolja];
    int resenje;
    float sum=0.0;
    
    x[1]=0;
    k=1;
    while (k > 0) {
        x[k]++;
        if (x[k]<=n){
            if (k == n) {
                resenje=0;
                for (ind=1;ind<=brPolja; ind++) {sum +=(float)x[ind];}
                                                //{sprintf(resenje, "%d", x[ind]);}
            }
            else {k++; x[k]=0;}
         }
         else {k--;}
    }
    //sum =(float)resenje;
    *output = sum;
}
