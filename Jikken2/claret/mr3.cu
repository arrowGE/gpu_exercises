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
void inter(float xj[3], float xi[3], float fi[3], int t, float xmax, float xmax1)//デバイス側計算部//VG_MATRIX *d_matrix, int t, float xmax, float xmax1)
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
  
  r     = sqrtf(dn2);
  inr   = 1.0f / r;    
  inr2  = inr  * inr;
  inr4  = inr2 *inr2;
  inr8  = inr4 * inr4;
  d3    = pb * d_matrix[t].pol * expf( (d_matrix[t].sigm - r) * d_matrix[t].ipotro);

  dphir = ( d3 * d_matrix[t].ipotro * inr
    - 6.0f * d_matrix[t].pc * inr8
    - 8.0f * d_matrix[t].pd * inr8 * inr2
    + inr2 * inr * d_matrix[t].zz );

  if(dn2==0.0) //if文簡略化
  {
    dphir=0.0;
  }

  for(k=0; k<3; k++) fi[k] += dphir * dr[k];
  
}

extern "C" __global__ 
void nacl_kernel_gpu(VG_XVEC *x, int n, int nat, float xmax, float *fvec)//デバイス側メイン//VG_MATRIX *d_matrix, float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k,js,nj;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;

  for (j = 0; j < n; j+=NTHRE)
  {
    if(j + NTHRE > n)//nがNTHREの倍数以外なら差分だけ計算する
    {
      nj = n - j;
    }
    else
    {
      nj = NTHRE;
    }
    __syncthreads();
    s_xj[tid] = x[j+tid];//シェアードメモリを使用
    __syncthreads();

    for(js = 0; js < nj; js++)
    {
      inter(s_xj[js].r, xi, fi, atypei + s_xj[js].atype, xmax, xmax1);
    }
    
    //ループアンローリング
    /*inter(s_xj[0].r, xi, fi, atypei + s_xj[0].atype, xmax, xmax1);
    inter(s_xj[1].r, xi, fi, atypei + s_xj[1].atype, xmax, xmax1);
    inter(s_xj[2].r, xi, fi, atypei + s_xj[2].atype, xmax, xmax1);
    inter(s_xj[3].r, xi, fi, atypei + s_xj[3].atype, xmax, xmax1);
    inter(s_xj[4].r, xi, fi, atypei + s_xj[4].atype, xmax, xmax1);
    inter(s_xj[5].r, xi, fi, atypei + s_xj[5].atype, xmax, xmax1);
    inter(s_xj[6].r, xi, fi, atypei + s_xj[6].atype, xmax, xmax1);
    inter(s_xj[7].r, xi, fi, atypei + s_xj[7].atype, xmax, xmax1);
    inter(s_xj[8].r, xi, fi, atypei + s_xj[8].atype, xmax, xmax1);
    inter(s_xj[9].r, xi, fi, atypei + s_xj[9].atype, xmax, xmax1);

    inter(s_xj[10].r, xi, fi, atypei + s_xj[10].atype, xmax, xmax1);
    inter(s_xj[11].r, xi, fi, atypei + s_xj[11].atype, xmax, xmax1);
    inter(s_xj[12].r, xi, fi, atypei + s_xj[12].atype, xmax, xmax1);
    inter(s_xj[13].r, xi, fi, atypei + s_xj[13].atype, xmax, xmax1);
    inter(s_xj[14].r, xi, fi, atypei + s_xj[14].atype, xmax, xmax1);
    inter(s_xj[15].r, xi, fi, atypei + s_xj[15].atype, xmax, xmax1);
    inter(s_xj[16].r, xi, fi, atypei + s_xj[16].atype, xmax, xmax1);
    inter(s_xj[17].r, xi, fi, atypei + s_xj[17].atype, xmax, xmax1);
    inter(s_xj[18].r, xi, fi, atypei + s_xj[18].atype, xmax, xmax1);
    inter(s_xj[19].r, xi, fi, atypei + s_xj[19].atype, xmax, xmax1);

    inter(s_xj[20].r, xi, fi, atypei + s_xj[20].atype, xmax, xmax1);
    inter(s_xj[21].r, xi, fi, atypei + s_xj[21].atype, xmax, xmax1);
    inter(s_xj[22].r, xi, fi, atypei + s_xj[22].atype, xmax, xmax1);
    inter(s_xj[23].r, xi, fi, atypei + s_xj[23].atype, xmax, xmax1);
    inter(s_xj[24].r, xi, fi, atypei + s_xj[24].atype, xmax, xmax1);
    inter(s_xj[25].r, xi, fi, atypei + s_xj[25].atype, xmax, xmax1);
    inter(s_xj[26].r, xi, fi, atypei + s_xj[26].atype, xmax, xmax1);
    inter(s_xj[27].r, xi, fi, atypei + s_xj[27].atype, xmax, xmax1);
    inter(s_xj[28].r, xi, fi, atypei + s_xj[28].atype, xmax, xmax1);
    inter(s_xj[29].r, xi, fi, atypei + s_xj[29].atype, xmax, xmax1);
    
    inter(s_xj[30].r, xi, fi, atypei + s_xj[30].atype, xmax, xmax1);
    inter(s_xj[31].r, xi, fi, atypei + s_xj[31].atype, xmax, xmax1);
    inter(s_xj[32].r, xi, fi, atypei + s_xj[32].atype, xmax, xmax1);
    inter(s_xj[33].r, xi, fi, atypei + s_xj[33].atype, xmax, xmax1);
    inter(s_xj[34].r, xi, fi, atypei + s_xj[34].atype, xmax, xmax1);
    inter(s_xj[35].r, xi, fi, atypei + s_xj[35].atype, xmax, xmax1);
    inter(s_xj[36].r, xi, fi, atypei + s_xj[36].atype, xmax, xmax1);
    inter(s_xj[37].r, xi, fi, atypei + s_xj[37].atype, xmax, xmax1);
    inter(s_xj[38].r, xi, fi, atypei + s_xj[38].atype, xmax, xmax1);
    inter(s_xj[39].r, xi, fi, atypei + s_xj[39].atype, xmax, xmax1);

    inter(s_xj[40].r, xi, fi, atypei + s_xj[40].atype, xmax, xmax1);
    inter(s_xj[41].r, xi, fi, atypei + s_xj[41].atype, xmax, xmax1);
    inter(s_xj[42].r, xi, fi, atypei + s_xj[42].atype, xmax, xmax1);
    inter(s_xj[43].r, xi, fi, atypei + s_xj[43].atype, xmax, xmax1);
    inter(s_xj[44].r, xi, fi, atypei + s_xj[44].atype, xmax, xmax1);
    inter(s_xj[45].r, xi, fi, atypei + s_xj[45].atype, xmax, xmax1);
    inter(s_xj[46].r, xi, fi, atypei + s_xj[46].atype, xmax, xmax1);
    inter(s_xj[47].r, xi, fi, atypei + s_xj[47].atype, xmax, xmax1);
    inter(s_xj[48].r, xi, fi, atypei + s_xj[48].atype, xmax, xmax1);
    inter(s_xj[49].r, xi, fi, atypei + s_xj[49].atype, xmax, xmax1);

    inter(s_xj[50].r, xi, fi, atypei + s_xj[50].atype, xmax, xmax1);
    inter(s_xj[51].r, xi, fi, atypei + s_xj[51].atype, xmax, xmax1);
    inter(s_xj[52].r, xi, fi, atypei + s_xj[52].atype, xmax, xmax1);
    inter(s_xj[53].r, xi, fi, atypei + s_xj[53].atype, xmax, xmax1);
    inter(s_xj[54].r, xi, fi, atypei + s_xj[54].atype, xmax, xmax1);
    inter(s_xj[55].r, xi, fi, atypei + s_xj[55].atype, xmax, xmax1);
    inter(s_xj[56].r, xi, fi, atypei + s_xj[56].atype, xmax, xmax1);
    inter(s_xj[57].r, xi, fi, atypei + s_xj[57].atype, xmax, xmax1);
    inter(s_xj[58].r, xi, fi, atypei + s_xj[58].atype, xmax, xmax1);
    inter(s_xj[59].r, xi, fi, atypei + s_xj[59].atype, xmax, xmax1);

    inter(s_xj[60].r, xi, fi, atypei + s_xj[60].atype, xmax, xmax1);
    inter(s_xj[61].r, xi, fi, atypei + s_xj[61].atype, xmax, xmax1);
    inter(s_xj[62].r, xi, fi, atypei + s_xj[62].atype, xmax, xmax1);
    inter(s_xj[63].r, xi, fi, atypei + s_xj[63].atype, xmax, xmax1);*/
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


