/* This code is modified by Tetsu Narumi
   from the exp2.cpp in http://homepage1.nifty.com/herumi/diary/0911.html
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <values.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

static float expTbl0[128 + 1];
#define d0MaskLen 12
static float expTbl1[1 << d0MaskLen];

#define NUM_OF_ARRAY(x) (sizeof(x) / sizeof(*x))
#define mask(x) ((1U << (x)) - 1)

union fi {
  float f;
  unsigned int i;
};

void InitExp()
{
  size_t i;
  for (i = 0; i < NUM_OF_ARRAY(expTbl0); i++)
  {
    expTbl0[i] = expf((float)i - 1);
  }
  for (i = 0; i < (1U << d0MaskLen); i++)
  {
    union fi fi;
    fi.i = (127 << 23) | (i << 11);
    expTbl1[i] = expf(fi.f);
  }
}

/*
  log(FLT_MAX) = 88.7 then return FLT_MAX if floor(x) >= 89
*/

float Exp(float x)
{
#if 1
  int isPositive = 0;
  if (x >= 0)
  {
    isPositive = 1;
  }
  else
  {
    x = -x;
    isPositive = 0;
  }
#endif
  int n = (int)floor(x) - 1;
  float e;

  if (n >= 88)
    e = FLT_MAX;
  else
  {
    float d = x - n;

    union fi fi;
    fi.f = d;
    unsigned int di = fi.i;

    /*
      sign + exponent + mantissa
      [1]      [8]      [12][11]
      d0  d1
    */
    int d0i = (di >> 11) & mask(12);

    fi.i = di & ((mask(8) << 23) | mask(11));
    float d1 = fi.f;
    e = expTbl0[n + 1] * expTbl1[d0i] * d1;
  }
#if 1
  if (!isPositive)
  {
    e = 1 / e;
  }
#endif
  return e;
}

//#define MAIN
#ifdef MAIN
void get_cputime(double *laptime, double *sprittime)
{
  struct timeval tv;
  struct timezone tz;
  double sec, microsec;

  gettimeofday(&tv, &tz);
  sec = tv.tv_sec;
  microsec = tv.tv_usec;

  *sprittime = sec + microsec * 1e-6 - *laptime;
  *laptime = sec + microsec * 1e-6;
}

void MR3calcnacl_CPU_expf(double x[], int n, int atype[], int nat,
                         double pol[], double sigm[], double ipotro[],
                         double pc[], double pd[], double zz[],
                         int tblno, double xmax, int periodicflag,
                         double force[])
{
  int i, j, k, t;
  double xmax1, dn2, r, inr, inr2, inr4, inr8, d3, dr[3], fi[3];
  double pb = 0.338e-19 / (14.39 * 1.60219e-19), dphir;
  if ((periodicflag & 1) == 0)
    xmax *= 2;
  xmax1 = 1.0 / xmax;
#pragma omp parallel for private(k, j, dn2, dr, r, inr, inr2, inr4, inr8, t, d3, dphir, fi)
  for (i = 0; i < n; i++)
  {
    for (k = 0; k < 3; k++)
      fi[k] = 0.0;
    for (j = 0; j < n; j++)
    {
      dn2 = 0.0;
      for (k = 0; k < 3; k++)
      {
        dr[k] = x[i * 3 + k] - x[j * 3 + k];
        dr[k] -= rint(dr[k] * xmax1) * xmax;
        dn2 += dr[k] * dr[k];
      }
      if (dn2 != 0.0)
      {
        r = sqrt(dn2);
        inr = 1.0 / r;
        inr2 = inr * inr;
        inr4 = inr2 * inr2;
        inr8 = inr4 * inr4;
        t = atype[i] * nat + atype[j];
        d3 = pb * pol[t] * expf((sigm[t] - r) * ipotro[t]);
        dphir = (d3 * ipotro[t] * inr - 6.0 * pc[t] * inr8 - 8.0 * pd[t] * inr8 * inr2 + inr2 * inr * zz[t]);
        for (k = 0; k < 3; k++)
          fi[k] += dphir * dr[k];
      }
    }
    for (k = 0; k < 3; k++)
      force[i * 3 + k] = fi[k];
  }
}

void MR3calcnacl_CPU_Exp(double x[], int n, int atype[], int nat,
                     double pol[], double sigm[], double ipotro[],
                     double pc[], double pd[], double zz[],
                     int tblno, double xmax, int periodicflag,
                     double force[])
{
  int i, j, k, t;
  double xmax1, dn2, r, inr, inr2, inr4, inr8, d3, dr[3], fi[3];
  double pb = 0.338e-19 / (14.39 * 1.60219e-19), dphir;
  if ((periodicflag & 1) == 0)
    xmax *= 2;
  xmax1 = 1.0 / xmax;
#pragma omp parallel for private(k, j, dn2, dr, r, inr, inr2, inr4, inr8, t, d3, dphir, fi)
  for (i = 0; i < n; i++)
  {
    for (k = 0; k < 3; k++)
      fi[k] = 0.0;
    for (j = 0; j < n; j++)
    {
      dn2 = 0.0;
      for (k = 0; k < 3; k++)
      {
        dr[k] = x[i * 3 + k] - x[j * 3 + k];
        dr[k] -= rint(dr[k] * xmax1) * xmax;
        dn2 += dr[k] * dr[k];
      }
      if (dn2 != 0.0)
      {
        r = sqrt(dn2);
        inr = 1.0 / r;
        inr2 = inr * inr;
        inr4 = inr2 * inr2;
        inr8 = inr4 * inr4;
        t = atype[i] * nat + atype[j];
        d3 = pb * pol[t] * Exp((sigm[t] - r) * ipotro[t]);
        dphir = (d3 * ipotro[t] * inr - 6.0 * pc[t] * inr8 - 8.0 * pd[t] * inr8 * inr2 + inr2 * inr * zz[t]);
        for (k = 0; k < 3; k++)
          fi[k] += dphir * dr[k];
      }
    }
    for (k = 0; k < 3; k++)
      force[i * 3 + k] = fi[k];
  }
}


int main()
{
  InitExp();
  const float e = 1e-6f;
  const int N = 10;
  float sum, x;
  double ltime, stime;
  int i;
  {
    sum = 0;
    float max = 0;
    int count = 0;
    for (x = 0; x <= 2; x += e)
    {
      float a = expf(x);
      float b = Exp(x);
      float diff = fabs(a - b);
      max = max(max, diff);
      sum += diff;
      count++;
      //		printf("x=%f, a=%e, b=%e\n", x, a, b);
    }
    printf("max=%e, ave=%e\n", max, sum / count);
  }
  {
    sum = 0;
    float max = 0;
    int count = 0;
    for (x = 0; x <= 2; x += e)
    {
      float a = expf(x);
      float b = Exp(x);
      float diff = fabs(a - b);
      max = max(max, diff);
      sum += diff;
      count++;
      //		printf("x=%f, a=%e, b=%e\n", x, a, b);
    }
    printf("max=%e, ave=%e\n", max, sum / count);
  }
  {
    sum = 0;
    get_cputime(&ltime, &stime);
    for (i = 0; i < N; i++)
    {
      for (x = 0; x <= 3; x += e)
      {
        sum += expf(x);
      }
    }
    get_cputime(&ltime, &stime);
    printf("sum=%f, org exp time=%f\n", sum / N, stime);
  }
  {
    sum = 0;
    get_cputime(&ltime, &stime);
    for (i = 0; i < N; i++)
    {
      for (x = 0; x <= 3; x += e)
      {
        sum += Exp(x);
      }
    }
    get_cputime(&ltime, &stime);
    printf("sum=%f, my  exp time=%f\n", sum / N, stime);
  }
}
#endif
