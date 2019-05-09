#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cutil.h>
#define CUDA_SAFE_CALL(x) (x);
#define CUT_CHECK_ERROR(x) ;

#define NMAX      8192
#define NTHRE       64  
#define ATYPE        8
#define ATYPE2    (ATYPE * ATYPE)

typedef struct {
  float r[3];
  int atype;
} VG_XVEC;

typedef struct {
  float pol;
  float sigm;
  float ipotro;
  float pc;
  float pd;
  float zz;
} VG_MATRIX;

__constant__ VG_MATRIX d_matrix[sizeof(VG_MATRIX)*2*2];

__device__ __inline__ 
void inter(float xj[3], float xi[3], float fi[3], 
  int t, float xmax, float xmax1)//VG_MATRIX *d_matrix, int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++)
  {
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  if(dn2 != 0.0f)
  {
    r     = sqrtf(dn2);
    inr   = 1.0f / r;
    inr2  = inr  * inr;
    inr4  = inr2 * inr2;
    inr8  = inr4 * inr4;
    d3    = pb * d_matrix[t].pol * expf( (d_matrix[t].sigm - r) * d_matrix[t].ipotro);
    dphir = ( d3 * d_matrix[t].ipotro * inr
	    - 6.0f * d_matrix[t].pc * inr8
	    - 8.0f * d_matrix[t].pd * inr8 * inr2
	    + inr2 * inr * d_matrix[t].zz );
    for(k=0; k<3; k++) fi[k] += dphir * dr[k];
  }
}

extern "C" __global__ 
void nacl_kernel_gpu(VG_XVEC *x, int n, int nat, float xmax, float *fvec)//VG_MATRIX *d_matrix, float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  for (j = 0; j < n; j++){
    inter(x[j].r, xi, fi, atypei + x[j].atype, xmax, xmax1);
  }
  if(i<n) for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}


extern "C"
void MR3calcnacl(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  //static VG_MATRIX *d_matrix=NULL;
  
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  static VG_XVEC   *vec=NULL;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak)
  {
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      //CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*nalloc*3));
      //CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(vec);
      if((vec=(VG_XVEC *)malloc(sizeof(VG_XVEC)*(nalloc+NTHRE)))==NULL)
      {
	      fprintf(stderr,"** error : can't malloc vec **\n");
	      exit(1);
      }
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL)
      {
	      fprintf(stderr,"** error : can't malloc forcef **\n");
	      exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++)
    {
      for(j=0;j<nat;j++)
      {
	      matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	      matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	      matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	      matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	      matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	      matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
    //CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));
    cudaMemcpyToSymbol(d_matrix,matrix,sizeof(VG_MATRIX)*2*2);
    n_bak=n;
  }

  for(i=0;i<(n+NTHRE-1)/NTHRE*NTHRE;i++)
  {
    if(i<n)
    {
      for(j=0;j<3;j++)
      {
	      vec[i].r[j]=x[i*3+j];
      }
      vec[i].atype=atype[i];
    }
    else
    {
      for(j=0;j<3;j++)
      {
	      vec[i].r[j]=0.0f;
      }
      vec[i].atype=0;
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*((n+NTHRE-1)/NTHRE*NTHRE),
  			    cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_gpu<<< grid, threads >>>(d_x,n,nat,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}


extern "C"
void MR3init(void)
{
}

extern "C"
void MR3free(void)
{
}


