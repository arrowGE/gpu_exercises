#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

static void get_cputime(double *laptime, double *sprittime)
{
  struct timeval tv;
  struct timezone tz;
  double sec,microsec;

  gettimeofday(&tv, &tz);
  sec=tv.tv_sec;
  microsec=tv.tv_usec;

  *sprittime = sec + microsec * 1e-6 - *laptime;
  *laptime = sec + microsec * 1e-6;
}

void MR3calcnacl_correct(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j,k,t;
  double xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  double pb=0.338e-19/(14.39*1.60219e-19),dphir; 
  if((periodicflag & 1)==0) xmax *= 2;
  xmax1 = 1.0 / xmax;
#pragma omp parallel for private(k,j,dn2,dr,r,inr,inr2,inr4,inr8,t,d3,dphir,fi)
  for(i=0; i<n; i++){
    for(k=0; k<3; k++) fi[k] = 0.0;
    for(j=0; j<n; j++){
      dn2 = 0.0;
      for(k=0; k<3; k++){
	dr[k] =  x[i*3+k] - x[j*3+k];
	dr[k] -= rint(dr[k] * xmax1) * xmax;
	dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0){
	r     = sqrt(dn2);
	inr   = 1.0  / r;
	inr2  = inr  * inr;
	inr4  = inr2 * inr2;
	inr8  = inr4 * inr4;
	t     = atype[i] * nat + atype[j];
	d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
	dphir = ( d3 * ipotro[t] * inr
		  - 6.0 * pc[t] * inr8
		  - 8.0 * pd[t] * inr8 * inr2
		  + inr2 * inr * zz[t] );
	for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}

void MR3calcnacl_CPU(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j,k,t;
  double xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  double pb=0.338e-19/(14.39*1.60219e-19),dphir; 
  if((periodicflag & 1)==0) xmax *= 2;
  xmax1 = 1.0 / xmax;
#pragma omp parallel for private(k,j,dn2,dr,r,inr,inr2,inr4,inr8,t,d3,dphir,fi)
  for(i=0; i<n; i++){
    for(k=0; k<3; k++) fi[k] = 0.0;
    for(j=0; j<n; j++){
      dn2 = 0.0;
      for(k=0; k<3; k++){
	dr[k] =  x[i*3+k] - x[j*3+k];
	dr[k] -= rint(dr[k] * xmax1) * xmax;
	dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0){
	r     = sqrt(dn2);
	inr   = 1.0  / r;
	inr2  = inr  * inr;
	inr4  = inr2 * inr2;
	inr8  = inr4 * inr4;
	t     = atype[i] * nat + atype[j];
	d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
	dphir = ( d3 * ipotro[t] * inr
		  - 6.0 * pc[t] * inr8
		  - 8.0 * pd[t] * inr8 * inr2
		  + inr2 * inr * zz[t] );
	for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}
    


int main(int argc, char **argv)
{
  int i,j,n,nstep=1,nat=2;
  double *x,*a1,*a2;
  double *pol,*sigm,*ipotro,*pc,*pd,*zz;
  int *atype;
  double xmax=100.0;
  double ltime,stime;
  double avr,aone,err,eone;
  double favr,fisize,eavr,eisize;

  if(argc!=3 && argc!=4){
    printf("usage : %s number_of_particles calc_mode (number_of_steps)\n",argv[0]);
    printf("  calc_mode : 0 -- original routine is used\n");
    printf("              1 -- GPU is used\n");
    return 1;
  }

  // set number of particles
  sscanf(argv[1],"%d",&n);
  printf("Number of particle is %d\n",n);

  // set number of steps
  if(argc==4){
    sscanf(argv[3],"%d",&nstep);
  }
  printf("Number of steps is %d\n",nstep);

  // allocate variables
  if((x=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc x **\n");
    return 1;
  }
  if((a1=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc a1 **\n");
    return 1;
  }
  if((a2=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc a2 **\n");
    return 1;
  }
  if((atype=(int *)malloc(sizeof(int)*n))==NULL){
    fprintf(stderr,"** error : can't malloc atype **\n");
    return 1;
  }
  if((pol=(double *)malloc(sizeof(double)*nat*nat))==NULL){
    fprintf(stderr,"** error : can't malloc pol **\n");
  }
  if((sigm=(double *)malloc(sizeof(double)*nat*nat))==NULL){
    fprintf(stderr,"** error : can't malloc sigm **\n");
  }
  if((ipotro=(double *)malloc(sizeof(double)*nat*nat))==NULL){
    fprintf(stderr,"** error : can't malloc ipotro **\n");
  }
  if((pc=(double *)malloc(sizeof(double)*nat*nat))==NULL){
    fprintf(stderr,"** error : can't malloc pc **\n");
  }
  if((pd=(double *)malloc(sizeof(double)*nat*nat))==NULL){
    fprintf(stderr,"** error : can't malloc pd **\n");
  }
  if((zz=(double *)malloc(sizeof(double)*nat*nat))==NULL){
    fprintf(stderr,"** error : can't malloc zz **\n");
  }

  // set positions and types
  for(i=0;i<n;i++){
    for(j=0;j<3;j++){
      x[i*3+j]=drand48()*xmax;
    }
    atype[i]=drand48()*nat;
  }

  // set parameters between atoms
  for(i=0;i<nat;i++){
    for(j=0;j<nat;j++){
      pol[i*nat+j]=1.0+drand48();
      sigm[i*nat+j]=2.0+drand48();
      ipotro[i*nat+j]=3.0+drand48();
      pc[i*nat+j]=5.0+drand48();
      pd[i*nat+j]=4.0+drand48();
      zz[i*nat+j]=-1.0+2.0*drand48();
    }
  }


  // calculation stars from here
  if(argv[2][0]=='1'){
    MR3calcnacl(x,n,atype,nat,pol,sigm,ipotro,pc,pd,zz,0,xmax,1,a2);//GPUの1回目を先に計算
  }

  get_cputime(&ltime,&stime);//計測開始
  // calc with target routine
  for(i=0;i<nstep;i++){
    switch(argv[2][0]){
    case '0':
      MR3calcnacl_CPU(x,n,atype,nat,pol,sigm,ipotro,pc,pd,zz,0,xmax,1,a2);
      if(i==0) printf("CPU original routine is used\n");
      break;
    case '1':
      MR3calcnacl(x,n,atype,nat,pol,sigm,ipotro,pc,pd,zz,0,xmax,1,a2);
      if(i==0) printf("GPU routine is used\n");
      break;
    default:
      fprintf(stderr,"** error : cal_mode=%c is not supported **\n",argv[2][0]);
      return 1;
    }
  }
  get_cputime(&ltime,&stime);//計測終了
  stime = stime / (double)nstep;//nstepで割って1回分の計算時間を求める
  
  double temp=(double)n;
  switch(argv[2][0]){//計算速度を出力
  case '0':
    //printf("CPU calculation time  = %f [s]\n",stime);
    printf("CPU calculation speed = %f [Gflops]\n",temp*temp*78/stime/1e9);
    break;
  case '1':
    //printf("GPU calculation time  = %f [s]\n",stime);
    printf("GPU calculation speed = %f [Gflops]\n",temp*temp*78/stime/1e9);
    break;
  default:
    break;
  }
  
  // check result
  MR3calcnacl_correct(x,n,atype,nat,pol,sigm,ipotro,pc,pd,zz,0,xmax,1,a1);


  // error analysis will be here
  favr=0.0;
  eavr=0.0;
  fisize=0.0;
  eisize=0.0;
  double *a_acc;
  if((a_acc=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc a_acc **\n");
    return 1;
  }
  for(i=0;i<n;i++){
    fisize=sqrt(a1[i*3]*a1[i*3]+a1[i*3+1]*a1[i*3+1]+a1[i*3+2]*a1[i*3+2]);
    favr+=fisize;
  }
  favr = favr / (double)n;

  for(i=0;i<n;i++){
    for(j=0;j<3;j++){
      a_acc[i*3+j] = (a1[i*3+j] - a2[i*3+j]) / favr;
    }
    
    eisize = sqrt(a_acc[i*3]*a_acc[i*3] + a_acc[i*3+1]*a_acc[i*3+1] + a_acc[i*3+2]*a_acc[i*3+2]);
    eavr += eisize;
  }
  eavr = eavr / (double)n;

  printf("GPU calculation error = %e\n",eavr);


  // deallocate variables
  free(x);
  free(a1);
  free(a2);
  free(atype);
  free(pol);
  free(sigm);
  free(ipotro);
  free(pc);
  free(pd);
  free(zz);

  free(a_acc);
  
  return 0;
}
