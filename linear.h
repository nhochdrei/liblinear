#ifndef LIBLINEAR_H
#define LIBLINEAR_H

#define LIBLINEAR_VERSION 247

#ifdef __cplusplus
#include <functional>
#include <memory>
#include <utility>
extern "C" {
#endif // __cplus_plus

extern int liblinear_version;
extern unsigned liblinear_threads;

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL, ONECLASS_SVM = 21 }; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;             /* stopping tolerance */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
	double nu;
	double *init_sol;
	int regularize_bias;
};

struct model
{
	struct parameter param;
	int nr_class;           /* number of classes */
	int nr_feature;
	double *w;
	int *label;             /* label of each class */
	double bias;
	double rho;             /* one-class SVM only */
};

struct model* train(const struct problem *prob, const struct parameter *param);
void find_parameters(const struct problem *prob, const struct parameter *param, int nr_fold, double start_C, double start_p, double *best_C, double *best_p, double *best_score, int (*callback)(long, void*), void* callbackData);
double predict(const model *model_, const feature_node *x);
double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
double get_decfun_bias(const struct model *model_, int label_idx);
double get_decfun_rho(const struct model *model_);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
int check_regression_model(const struct model *model);
int check_oneclass_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));
void set_random_function(int (*random_func) (void));

#ifdef __cplusplus
}

namespace liblinear {
    namespace impl {
        struct model_destroyer_t
        {
            void operator()(model* mdl) const noexcept
            {
                free_and_destroy_model(&mdl);
            }
        };
    }
    using model_ptr_t = std::unique_ptr<model, impl::model_destroyer_t>;

    model_ptr_t train(const problem *prob, const parameter *param);
    void find_parameters(const problem *prob, const parameter *param, int nr_fold, double start_C, double start_p, double *best_C, double *best_p, double *best_score, int (*callback)(long, void*), void* callbackData);
    template<typename Callback>
    void find_parameters(const problem *prob, const parameter *param, int nr_fold, double start_C, double start_p, double *best_C, double *best_p, double *best_score, Callback&& callback)
    {
        using ft = std::function<bool(long)>;
        ft bound{std::forward<Callback>(callback)};
        ::find_parameters(prob, param, nr_fold, start_C, start_p, best_C, best_p, best_score, [](long it, void* data) { return (*static_cast<ft*>(data))(it) ? 1 : 0; }, &bound);
    }
    double predict(const model *model_, const feature_node *x);
    model *load_model(const char *model_file_name);

    inline namespace c_interface {
        using ::predict_values;
        using ::get_nr_feature;
        using ::get_nr_class;
        using ::get_labels;
        using ::get_decfun_coef;
        using ::get_decfun_bias;
        using ::save_model;
        using ::free_model_content;
        using ::free_and_destroy_model;
        using ::destroy_param;
        using ::check_parameter;
        using ::check_probability_model;
        using ::check_regression_model;
        using ::set_print_string_function;
        using ::set_random_function;
    }
}
#endif

#endif /* LIBLINEAR_H */

