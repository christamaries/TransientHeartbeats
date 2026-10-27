/*-----------------------------------------------------------------------------
Program Name: MEPSA (multi-excess peak search algorithm)
Author: Cristiano Guidorzi @ University of Ferrara, Italy
Version: 1.0
Date: October 2014

Purpose
Search an input time series for peaks.
Input time series is supposed to be background-subtracted (or just detrended),
evenly spaced, affected by uncorrelated Gaussian noise.

Usage
3 arguments are required:
   1. input time series (3 columns: time, rate, error on rate)
   2. excess patterns' file
   3. maximum rebin factor to be searched

Argument 2 is provided together with the present C code.
This could be changed by users themselves according to their need. In this
case, we recommend preliminary thorough testing before trusting the results.
Example:

mepsa input_lc.dat ~/MEPSA/excess_pattern_MEPSA_v0.dat 512 > mepsa.out

-----------------------------------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define MAX 150
#define REBTHRE 1
#define MAX_FINAL_PEAKS_N 1000

void rebin();
void  find_local_sigma_adiac_peak_asym();
int cmp_pos();

double final_peak_t[MAX_FINAL_PEAKS_N];
int    final_peak_c;

main(int argc, char *argv[])
{
  FILE *in, *sigma_file;
  int i,n,j, dimbs, dimreb, reb, first, ifirst, rebbs,dimrebbs, n_peak;
  char d[MAX], dsigma[MAX],found, *local_peak_flag;
  double *tempo,sigvar, *conteggi_bs, *errori_bs,ts,tsm,tsp,Btemp,snr,sum;
  double bintime, *variance, *variance_reb; 
  double *creb_bs, *ereb_bs, *treb, *creb, *treb_bs;
  double *sigma_thr;
  double peak_t, peak_c, peak_e, peak_reb;
  double prec_peak_t, prec_peak_c, prec_peak_e, prec_peak_reb;
  int    peak_rebin_f, prec_peak_rebin_f, max_rebin_factor, k, n_adiac,n_adiac_pre,n_adiac_post;
  double *local_valley_right, *local_valley_left;
  char   found_right, found_left, declassed;
  double final_bint[MAX_FINAL_PEAKS_N];
  double final_peak[MAX_FINAL_PEAKS_N],final_epeak[MAX_FINAL_PEAKS_N];
  int    final_peak_rebin[MAX_FINAL_PEAKS_N],final_peak_phase[MAX_FINAL_PEAKS_N];
  int    final_peak_nadiac[MAX_FINAL_PEAKS_N];
  int    final_peak_criteria[MAX_FINAL_PEAKS_N];
  int    m, good_final, *pos, n_overlapping, n_criteria;

  in=fopen(argv[1],"r");
 
  if(argc!=5)
    {
      puts("\nProgram Name: MEPSA (multi-excess peak search algorithm)");
      puts("Author: Cristiano Guidorzi @ University of Ferrara, Italy");
      puts("Version: 1.0\nDate: October 2014\n");

      puts("Purpose\nSearch an input time series for peaks.");
      puts("Input time series is supposed to be background-subtracted (or just detrended),");
      puts("evenly spaced, affected by uncorrelated Gaussian noise.\n");

      puts("Usage\n3 arguments are required:");
      puts("1. input time series (3 columns: time, rate, error on rate)");
      puts("2. excess patterns' file");
      puts("3. maximum rebin factor to be searched\n");
      puts("4. name of the file to be saved with the results\n");

      puts("Argument 2 is provided together with the present C code.");
      puts("This could be changed by users themselves according to their need. In this");
      puts("case, we recommend preliminary thorough testing before trusting the results.");
      puts("Example:\n");

      fprintf(stdout,"%s input_lc.dat ~/MEPSA/excess_pattern_MEPSA_v0.dat 512 > mepsa.out\n\n",argv[0]);

      exit(1);
    }

  if(in==NULL)
    {
      printf("File called \"%s\" not found\n",argv[1]);
      exit(1);
    }

  sigma_file = fopen(argv[2], "r");
  if(!sigma_file) {
    fprintf(stdout, "Cannot find file %s . Exit program.\n",argv[2]);
    exit(1);
  }

  sscanf(argv[3], "%d", &max_rebin_factor);
  


  i=0;
  while(!feof(in))
    {
      if(fgets(d,MAX-1,in)==NULL)
        break;
      ++i;
    }

  n=i;i=0;
  /* printf("Dim file input: %d\n",n); */
  tempo=(double*)malloc(n*sizeof(double));
  conteggi_bs=(double*)malloc(n*sizeof(double));
  errori_bs=(double*)malloc(n*sizeof(double));
  variance=(double*)malloc(n*sizeof(double));

  if (tempo==NULL)
    {
      printf("Niente memoria disponibile per il tempo: \n");
      exit(1);    
    }
  if ((conteggi_bs==NULL)||(errori_bs==NULL))
    {
      printf("Niente memoria disponibile per i conteggi: \n");
      exit(1);    
    }

 
  rewind(in);
  
  while(!feof(in))
    {
      if (fgets(d,MAX-1,in)==NULL)
	break;
      sscanf(d,"%lg %lg %lg",(tempo+i),(conteggi_bs+i),(errori_bs+i));
      *(variance+i) = *(errori_bs+i)*(*(errori_bs+i));
      ++i;
    }
  fclose(in);
  
  
  bintime = (*(tempo + 1))-(*(tempo));

  

  final_peak_c = 0;
  n_criteria=0;

  while(!feof(sigma_file)) {
    if(fgets(dsigma,MAX-1,sigma_file)==NULL) break;
    if(dsigma[0]=='#')  continue;
    ++n_criteria;
    sscanf(dsigma, "%d %d", &n_adiac_pre, &n_adiac_post);
    //printf("n_adiac: %d\n",n_adiac_pre,n_adiac_post);
    n_adiac=n_adiac_pre + n_adiac_post;
    sigma_thr = (double *) malloc(sizeof(double)*n_adiac);
    switch(n_adiac) {
    case 0 : printf("Cannot accept null value for sigma_thr.\nExit program.\n"); exit(1);
    case 1 : sscanf(dsigma, "%*s %*s %lg", (sigma_thr+0)); break;
    case 2 : sscanf(dsigma, "%*s %*s %lg %lg", (sigma_thr+0), (sigma_thr+1)); break;
    case 3 : sscanf(dsigma, "%*s %*s %lg %lg %lg", (sigma_thr+0), (sigma_thr+1), (sigma_thr+2)); break;
    case 4 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),(sigma_thr+2),(sigma_thr+3)); break;
    case 5 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),(sigma_thr+2),(sigma_thr+3),\
		    (sigma_thr+4)); break;
    case 6 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),(sigma_thr+2),\
		    (sigma_thr+3),(sigma_thr+4),(sigma_thr+5)); break;
    case 7 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),(sigma_thr+2),\
		    (sigma_thr+3),(sigma_thr+4),(sigma_thr+5),(sigma_thr+6)); break;
    case 8 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),(sigma_thr+2),\
		    (sigma_thr+3),(sigma_thr+4),(sigma_thr+5),(sigma_thr+6),(sigma_thr+7)); break;
    case 9 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),\
		    (sigma_thr+2),(sigma_thr+3),(sigma_thr+4),(sigma_thr+5),(sigma_thr+6),(sigma_thr+7),\
		    (sigma_thr+8)); break;
    case 10 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),\
		     (sigma_thr+2),(sigma_thr+3),(sigma_thr+4),(sigma_thr+5),(sigma_thr+6),(sigma_thr+7),\
		     (sigma_thr+8),(sigma_thr+9)); break;
    case 11 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),\
		     (sigma_thr+2),(sigma_thr+3),(sigma_thr+4),(sigma_thr+5),(sigma_thr+6),(sigma_thr+7),\
		     (sigma_thr+8),(sigma_thr+9),(sigma_thr+10)); break;
    case 12 : sscanf(dsigma, "%*s %*s %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",(sigma_thr+0),(sigma_thr+1),\
		     (sigma_thr+2),(sigma_thr+3),(sigma_thr+4),(sigma_thr+5),(sigma_thr+6),(sigma_thr+7),\
		     (sigma_thr+8),(sigma_thr+9),(sigma_thr+10),(sigma_thr+11)); break;
    default : printf("Cannot accept >12 for sigma_thr.\nExit program.\n"); exit(1);
    }

    sum=0.0;
    j = 0;
    
    found = 0;
    reb = 1;
    do
      {
	rebbs = (reb < REBTHRE) ? REBTHRE : reb;
	for(k=0; k<reb; ++k) {
	  dimreb = (n-k)/reb;
	  dimrebbs = (n-k) / rebbs;

	  treb_bs=(double*)malloc(dimrebbs*sizeof(double));
	  creb_bs=(double*)malloc(dimrebbs*sizeof(double));
	  ereb_bs=(double*)malloc(dimrebbs*sizeof(double));
	  variance_reb=(double*)malloc(dimrebbs*sizeof(double));
	  local_peak_flag=(char*)malloc(dimrebbs*sizeof(char));
	  local_valley_left=(double*)malloc(dimrebbs*sizeof(double));
	  local_valley_right=(double*)malloc(dimrebbs*sizeof(double));
	  
	  if((creb_bs==NULL)||(ereb_bs==NULL)||(treb_bs==NULL)) {
	    printf("Niente memoria disponibile per i conteggi rebinnati: \n");
	    exit(1);    
	  }
	  
	  // rebin(tempo+k, treb_bs, n-k, rebbs);
	  for(i=0; i<dimrebbs; ++i) *(treb_bs+i) = *(tempo+k+i*rebbs); // newbin start time
	  rebin(conteggi_bs+k, creb_bs, n-k, rebbs);
	  rebin(variance+k, variance_reb, n-k, rebbs);
	  for(i=0; i<dimrebbs; ++i) *(ereb_bs+i) = sqrt(*(variance_reb+i));
	  
	  find_local_sigma_adiac_peak_asym(creb_bs, ereb_bs, dimrebbs, \
				      sigma_thr, n_adiac_pre, n_adiac_post, local_peak_flag);
	  
	  
	  n_peak=0;
	  for(i=0; i<dimrebbs; ++i)
	    if((*(local_peak_flag+i))) ++n_peak;
	  if(n_peak) {
	    n_peak = 0;
	    for(i=0; i<dimrebbs; ++i) {
	      if(!(*(local_peak_flag+i)))  continue;
	      else {
		good_final = 1;
		++n_peak;
		if(final_peak_c) {
		  n_overlapping=0;
		  for(m=0; m<final_peak_c; ++m)
		    {
		      if(fabs(final_peak_t[m]+final_bint[m]/2.0-*(treb_bs+i)-reb*bintime/2.0)<(final_bint[m]+reb*bintime))
			  ++n_overlapping;
		    }
		  if(n_overlapping>1) good_final=0;
		  else if(n_overlapping==1) 
		    for(m=0; m<final_peak_c; ++m)
		      {
			if(fabs(final_peak_t[m]+final_bint[m]/2.0-*(treb_bs+i)-reb*bintime/2.0)<(final_bint[m]+reb*bintime)) {
			  if(*(creb_bs+i)/reb > final_peak[m]) {
			    final_peak_t[m] = *(treb_bs+i);
			    final_bint[m] = reb*bintime;
			    final_peak[m] = *(creb_bs+i)/reb;
			    final_epeak[m] = *(ereb_bs+i)/reb;
			    final_peak_rebin[m] = reb;
			    final_peak_phase[m] = k;
			    final_peak_nadiac[m] = n_adiac;
			    final_peak_criteria[m] = n_criteria;
			  }
			  good_final = 0;
			  break;
			}
		      }
		}
		if(good_final) {
		  final_peak_t[final_peak_c] = *(treb_bs+i);
		  final_bint[final_peak_c] = reb*bintime;
		  final_peak[final_peak_c] = *(creb_bs+i)/reb;
		  final_epeak[final_peak_c] = *(ereb_bs+i)/reb;
		  final_peak_rebin[final_peak_c] = reb;
		  final_peak_phase[final_peak_c] = k;
		  final_peak_nadiac[final_peak_c] = n_adiac;
		  final_peak_criteria[final_peak_c] = n_criteria;
		  ++final_peak_c;
		}
	      }
	    } 
	  }
	  free(creb_bs);  free(ereb_bs);  free(treb_bs);  free(variance_reb);
	  free(local_peak_flag); free(local_valley_left); free(local_valley_right);
	}
	//     if(found)  break;
	++reb;
	// fprintf(stdout,"reb: %d\n",reb);
      } while(reb<max_rebin_factor);
    free(sigma_thr);
  }
  
  fclose(sigma_file);

  free(tempo);
  free(conteggi_bs);
  free(errori_bs);
  
  // WRITE ON STANDARD OUTPUT
  if(final_peak_c) {
    pos = (int *) malloc(sizeof(int)*final_peak_c);
    for(i=0; i<final_peak_c; ++i) *(pos+i) = i;
    k = final_peak_c;
    qsort(pos, k, sizeof(int), &cmp_pos);
    fprintf(stdout,"#Peak RebF BinPhase\tPeakT     BinT\t   PeakR    EPeakR     SNR\tCriterium Nadiac\n");
    for(i=0; i<final_peak_c; ++i) {
      fprintf(stdout,"%3d  %3d  %3d\t%10.3f %8.3f\t%10.5f %9.5f  %6.2f\t%2d %2d\n",i+1,\
	      final_peak_rebin[pos[i]],\
	      final_peak_phase[pos[i]],final_peak_t[pos[i]]+final_bint[pos[i]]/2.0,final_bint[pos[i]],\
	      final_peak[pos[i]],final_epeak[pos[i]],final_peak[pos[i]]/final_epeak[pos[i]],\
	      final_peak_criteria[pos[i]], final_peak_nadiac[pos[i]]);
    }
  }
  
  // WRITE ON FILE
  int save_file=1;
  if(save_file){
    FILE *fptr;
    fptr = fopen(argv[4],"w");
    if(fptr==NULL){
      printf("ERROR in the definition of the pointer to the file; closing...!");   
      exit(1);             
    }
    if(final_peak_c) {
      pos = (int *) malloc(sizeof(int)*final_peak_c);
      for(i=0; i<final_peak_c; ++i) *(pos+i) = i;
      k = final_peak_c;
      qsort(pos, k, sizeof(int), &cmp_pos);
      fprintf(fptr,"#Peak RebF BinPhase\tPeakT     BinT\t   PeakR    EPeakR     SNR\tCriterium Nadiac\n");
      for(i=0; i<final_peak_c; ++i) {
        fprintf(fptr,"%3d  %3d  %3d\t%10.3f %8.3f\t%10.5f %9.5f  %6.2f\t%2d %2d\n",i+1,\
          final_peak_rebin[pos[i]],\
          final_peak_phase[pos[i]],final_peak_t[pos[i]]+final_bint[pos[i]]/2.0,final_bint[pos[i]],\
          final_peak[pos[i]],final_epeak[pos[i]],final_peak[pos[i]]/final_epeak[pos[i]],\
          final_peak_criteria[pos[i]], final_peak_nadiac[pos[i]]);
      }
    }
    fclose(fptr);
  }

}
/***********************************************************************/

void  rebin(double *x, double *y, int d, int reb)

{
  int     i, j=0, k=0;
  double  temp;

  for(i=0; i<d/reb; ++i)  *(y + i) = 0.0;
  temp = 0.0;
  k = 0;    j = 0;

  for(i=0; i<d; ++i)
    {
      temp += *(x + i);
      ++k;
      if(k==reb)
        {
          *(y + j++) = temp;
          k = 0;
          temp = 0.0;
        }
    }
}

/***********************************************************************/
void find_local_sigma_adiac_peak_asym(double *x, double *e, int n, double *sig, int n_adiac_pre,\
				      int n_adiac_post, char *flag)

{
  int     i, n_m, ret, j, first, n_adiac;
  double  m, p,f,c, ep, ef, ec, eval, excess, max;


  n_adiac = n_adiac_pre + n_adiac_post;

  for(i=n_adiac_pre; i<n-n_adiac_post; ++i) {
      c = *(x+i);
      ec = *(e+i);
      for(j=1; j<=n_adiac_pre; ++j) {
	p = *(x+i-n_adiac_pre+j-1);
	ep = *(e+i-n_adiac_pre+j-1);
	excess = (c-p)/sqrt(ec*ec+ep*ep);
	*(flag+i) = (excess >= *(sig+j-1)) ? 1 : 0;
	if(!(*(flag+i))) break;
      }
      if((*(flag+i)))
	{
	  for(j=1; j<=n_adiac_post; ++j) {
	    f = *(x+i+j);
	    ef = *(e+i+j);
	    excess = (c-f)/sqrt(ec*ec+ef*ef);
	    *(flag+i) = (excess >= *(sig+j-1+n_adiac_pre)) ? 1 : 0;
	    if(!(*(flag+i))) break;
	  }
	}
  }

  for(i=0; i<n_adiac_pre; ++i)     *(flag+i) = 0;
  for(i=n-n_adiac_post; i<n; ++i) *(flag+i) = 0;
}

/*******************************************************************/
int cmp_pos(const void *a, const void *b)
{
  double v1, v2;
  size_t i1, j1, i2, j2;
  int* i = (int*) a;
  int* j = (int*) b;

  v1 = final_peak_t[*i];
  v2 = final_peak_t[*j];

  if(v1 < v2) return -1;
  else if(v1 > v2) return 1;
  else return 0;
}
/*******************************************************************/
