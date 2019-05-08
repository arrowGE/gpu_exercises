#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cutil.h>
#define CUDA_SAFE_CALL(x) (x);
#define CUT_CHECK_ERROR(x) ;

#define D2F_AND_COPY(n,host_mem,device_mem,float_mem) \
  for(int i=0;i<(n);i++) ((float *)(float_mem))[i]=(host_mem)[i];\
  CUDA_SAFE_CALL(cudaMalloc((void **)&(device_mem),sizeof(float)*(n)));\
  CUDA_SAFE_CALL(cudaMemcpy((device_mem),(float_mem),sizeof(float)*(n),cudaMemcpyHostToDevice));

extern "C"
void MR3init(void)
{
}

extern "C"
void MR3free(void)
{
}

extern "C" __global__ 
void nacl_kernel(float *x, int n, int *atype, int nat, float *pol, float *sigm, float *ipotro,
		 float *pc, float *pd, float *zz, int tblno, float xmax, int periodicflag, 
		 float *force)
{
  /*
  x 原子の座標
  n 粒子数
  atype ある粒子の原子種類 0=Na 1=Cl
  nat 原子の種類の数 2
  pol,sigmo,iporto,pc,pd,zz nat*natの大きさを持つ配列 Na-Na=0 Na-Cl or Cl-Na = 1 or 2 Cl-Cl=3
  tblno 未使用
  xmax 周期的境界条件におけるセルの大きさ
  periodicflag 周期境界条件の時は1、それ以外は0 デフォルトでは0
  force force[i*3+0~2]で粒子iに働く力の大きさのx~z成分
  */

  int i,j,k,t;
  float xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir; 

  int js;
  __shared__ float s_xj[64*3];
  __shared__ int s_atypej[64];
  int tid = threadIdx.x;
  

  if((periodicflag & 1)==0) xmax *= 2.0f;
  xmax1 = 1.0f / xmax;//xmax1 = xmaxの逆数

  i = blockIdx.x * 64 + threadIdx.x;//スレッド番号取得

  if(i<n)
  {
    for(k=0; k<3; k++) fi[k] = 0.0f;

    for(j=0; j<n; j+=64)
    {
      __syncthreads();
      s_xj[tid*3+0] = x[(j+tid)*3+0];
      s_xj[tid*3+1] = x[(j+tid)*3+1];
      s_xj[tid*3+2] = x[(j+tid)*3+2];
      s_atypej[tid] = atype[j+tid];
      __syncthreads();
      for(js=0;js<64;js++)
      {
        dn2 = 0.0f;
        for(k=0; k<3; k++)
        {
          dr[k] =  x[i*3+k] - s_xj[js*3+k];
          dr[k] -= rintf(dr[k] * xmax1) * xmax;
          dn2   += dr[k] * dr[k];
        }
        if(dn2 != 0.0f)
        {
          r     = sqrtf(dn2);
          inr   = 1.0f  / r;
          inr2  = inr  * inr;
          inr4  = inr2 * inr2;
          inr8  = inr4 * inr4;
          
          t     = atype[i] * nat + s_atypej[js];//分子の組み合わせを判定 Na-Na=0 Na-Cl or Cl-Na = 1 or 2 Cl-Cl=3
          
          d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
          
          dphir = ( d3 * ipotro[t] * inr
              - 6.0f * pc[t] * inr8
              - 8.0f * pd[t] * inr8 * inr2
              + inr2 * inr * zz[t] );
              
          for(k=0; k<3; k++) fi[k] += dphir * dr[k];
        }
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}

extern "C" __global__ 
void nacl_kernel_original(float *x, int n, int *atype, int nat, float *pol, float *sigm, float *ipotro,
		 float *pc, float *pd, float *zz, int tblno, float xmax, int periodicflag, 
		 float *force)
{
  int i,j,k,t;
  float xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir; 

  if((periodicflag & 1)==0) xmax *= 2.0f;
  xmax1 = 1.0f / xmax;
  i = blockIdx.x * 64 + threadIdx.x;
  if(i<n)
  {
    for(k=0; k<3; k++) fi[k] = 0.0f;
    for(j=0; j<n; j++)
    {
      dn2 = 0.0f;
      for(k=0; k<3; k++)
      {
	      dr[k] =  x[i*3+k] - x[j*3+k];
	      dr[k] -= rintf(dr[k] * xmax1) * xmax;
	      dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0f)
      {
	      r     = sqrtf(dn2);
	      inr   = 1.0f  / r;
	      inr2  = inr  * inr;
	      inr4  = inr2 * inr2;
	      inr8  = inr4 * inr4;
	      t     = atype[i] * nat + atype[j];
	      d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
	      dphir = ( d3 * ipotro[t] * inr
		        - 6.0f * pc[t] * inr8
		        - 8.0f * pd[t] * inr8 * inr2
		        + inr2 * inr * zz[t] );
	      for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}

extern "C"
void MR3calcnacl(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,*d_atype;
  float *d_x,*d_pol,*d_sigm,*d_ipotro,*d_pc,*d_pd,*d_zz,*d_force,xmaxf=xmax;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(float)*nat*nat)
  {
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }

  // allocate global memory and copy from host to GPU
  D2F_AND_COPY(n*3,x,d_x,force);
  D2F_AND_COPY(nat*nat,pol,d_pol,force);
  D2F_AND_COPY(nat*nat,sigm,d_sigm,force);
  D2F_AND_COPY(nat*nat,ipotro,d_ipotro,force);
  D2F_AND_COPY(nat*nat,pc,d_pc,force);
  D2F_AND_COPY(nat*nat,pd,d_pd,force);
  D2F_AND_COPY(nat*nat,zz,d_zz,force);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_atype,sizeof(int)*n));
  CUDA_SAFE_CALL(cudaMemcpy(d_atype,atype,sizeof(int)*n,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*n*3));

  // call GPU kernel
  dim3 threads(64);
  dim3 grid((n+63)/64);
  nacl_kernel<<< grid, threads >>>(d_x,n,d_atype,nat,d_pol,d_sigm,d_ipotro,
				   d_pc,d_pd,d_zz,tblno,xmaxf,periodicflag,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(force,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=n*3-1;i>=0;i--) force[i]=((float *)force)[i];

  // free allocated global memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_atype));
  CUDA_SAFE_CALL(cudaFree(d_pol));
  CUDA_SAFE_CALL(cudaFree(d_sigm));
  CUDA_SAFE_CALL(cudaFree(d_ipotro));
  CUDA_SAFE_CALL(cudaFree(d_pc));
  CUDA_SAFE_CALL(cudaFree(d_pd));
  CUDA_SAFE_CALL(cudaFree(d_zz));
  CUDA_SAFE_CALL(cudaFree(d_force));
}
