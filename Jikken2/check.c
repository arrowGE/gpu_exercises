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


int main(int argc, char **argv)
{
  int i,j,n,nstep=1,nat=2;
  double *x,*a1,*a2;
  double *pol,*sigm,*ipotro,*pc,*pd,*zz;
  int *atype;
  double xmax=100.0;
  double ltime,stime;
  double avr,aone,err,eone;

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

  // check result
  MR3calcnacl_correct(x,n,atype,nat,pol,sigm,ipotro,pc,pd,zz,0,xmax,1,a1);


  // error analysis will be here
  favr=0.0;
  for(i=0;i<n;i++){
    fisize=sqrt(a1[i*3]*a1[i*3]+a1[i*3+1]*a1[i*3+1]+a1[i*3+2]*a1[i*3+2]);
    
  }


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
  
  return 0;
}
