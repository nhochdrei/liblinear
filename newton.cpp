#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "newton.h"

#include <memory>

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

// On entry *f must be the function value of w
// On exit w is updated and *f is the new function value
double function::linesearch_and_update(double *w, double *s, double *f, double *g, double alpha)
{
	double gTs = 0;
	double eta = 0.01;
	int n = get_nr_variable();
	int max_num_linesearch = 20;
	std::unique_ptr<double[]> w_new{new double[n]};
	double fold = *f;

	for (int i=0;i<n;i++)
		gTs += s[i] * g[i];

	int num_linesearch = 0;
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		for (int i=0;i<n;i++)
			w_new[i] = w[i] + alpha*s[i];
		*f = fun(w_new.get());
		if (*f - fold <= eta * alpha * gTs)
			break;
		else
			alpha *= 0.5;
	}

	if (num_linesearch >= max_num_linesearch)
	{
		*f = fold;
		return 0;
	}
	else
		memcpy(w, w_new.get(), sizeof(double)*n);
	
	return alpha;
}

void NEWTON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*newton_print_string)(buf);
}

NEWTON::NEWTON(const function *fun_obj, double eps, double eps_cg, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->eps_cg=eps_cg;
	this->max_iter=max_iter;
	newton_print_string = default_print;
}

NEWTON::~NEWTON()
{
}

void NEWTON::newton(double *w)
{
	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double step_size;
	double f, fold, actred;
	double init_step_size = 1;
	int search = 1, iter = 1, inc = 1;
	std::unique_ptr<double[]> s_{new double[n]};
	double* s = s_.get();
	std::unique_ptr<double[]> r_{new double[n]};
	double* r = r_.get();
	std::unique_ptr<double[]> g_{new double[n]};
	double* g = g_.get();

	const double alpha_pcg = 0.01;
	std::unique_ptr<double[]> M_{new double[n]};
	double* M = M_.get();

	// calculate gradient norm at w=0 for stopping condition.
	std::unique_ptr<double[]> w0_{new double[n]};
	double* w0 = w0_.get();
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	double gnorm0 = dnrm2_(&n, g, &inc);

	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	double gnorm = dnrm2_(&n, g, &inc);
	info("init f %5.3e |g| %5.3e\n", f, gnorm);

	if (gnorm <= eps*gnorm0)
		search = 0;

	while (iter <= max_iter && search)
	{
		fun_obj->get_diag_preconditioner(M);
		for(i=0; i<n; i++)
			M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
		cg_iter = pcg(g, M, s, r);

		fold = f;
		step_size = fun_obj->linesearch_and_update(w, s, &f, g, init_step_size);

		if (step_size == 0)
		{
			info("WARNING: line search fails\n");
			break;
		}

		fun_obj->grad(w, g);
		gnorm = dnrm2_(&n, g, &inc);

		info("iter %2d f %5.3e |g| %5.3e CG %3d step_size %4.2e \n", iter, f, gnorm, cg_iter, step_size);
		
		if (gnorm <= eps*gnorm0)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		actred = fold - f;
		if (fabs(actred) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred too small\n");
			break;
		}

		iter++;
	}

	if(iter >= max_iter)
		info("\nWARNING: reaching max number of Newton iterations\n");
}

int NEWTON::pcg(double *g, double *M, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	std::unique_ptr<double[]> d_{new double[n]};
	double* d = d_.get();
	std::unique_ptr<double[]> Hd_{new double[n]};
	double* Hd = Hd_.get();
	double zTr, znewTrnew, alpha, beta, cgtol, dHd;
	std::unique_ptr<double[]> z_{new double[n]};
	double* z = z_.get();
	double Q = 0, newQ, Qdiff;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		z[i] = r[i] / M[i];
		d[i] = z[i];
	}

	zTr = ddot_(&n, z, &inc, r, &inc);
	double gMinv_norm = sqrt(zTr);
	cgtol = min(eps_cg, sqrt(gMinv_norm));
	int cg_iter = 0;
	int max_cg_iter = max(n, 5);

	while (cg_iter < max_cg_iter)
	{
		cg_iter++;

		fun_obj->Hv(d, Hd);
		dHd = ddot_(&n, d, &inc, Hd, &inc);
		// avoid 0/0 in getting alpha
		if (dHd <= 1.0e-16)
			break;
		
		alpha = zTr/dHd;
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);

		// Using quadratic approximation as CG stopping criterion
		newQ = -0.5*(ddot_(&n, s, &inc, r, &inc) - ddot_(&n, s, &inc, g, &inc));
		Qdiff = newQ - Q;
		if (newQ <= 0 && Qdiff <= 0)
		{
			if (cg_iter * Qdiff >= cgtol * newQ)
				break;
		}
		else
		{
			info("WARNING: quadratic approximation > 0 or increasing in CG\n");
			break;
		}
		Q = newQ;

		for (i=0; i<n; i++)
			z[i] = r[i] / M[i];
		znewTrnew = ddot_(&n, z, &inc, r, &inc);
		beta = znewTrnew/zTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, z, &inc, d, &inc);
		zTr = znewTrnew;
	}

	if (cg_iter == max_cg_iter)
		info("WARNING: reaching maximal number of CG steps\n");

	return cg_iter;
}

void NEWTON::set_print_string(void (*print_string) (const char *buf))
{
	newton_print_string = print_string;
}
