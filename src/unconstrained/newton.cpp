//
// Created by lei on 22-11-1.
//

#include "../../include/unconstrained/newton.h"

namespace optim {
    using namespace geometry;

    template<typename vType>
    bool newton(Vector<vType> &init_vals,
                std::function<vType(const Vector<vType> &in_vec, Vector<vType> &grad, Matrix<vType> &hess,
                                    void *opt_data)> opt_obj_fn,
                void *opt_data,
                settings *set) {
        bool success = false;

        const size_t rows = init_vals.rows();

        const size_t max_iters = set->max_iters;
        const double grad_err = set->grad_err;

        Vector<vType> x = init_vals;
        Vector<vType> x_p = x;                                  // 用于更新 x 的临时值
        Vector<vType> grad(rows);                               // gradient vector
        Matrix<vType> hess(rows, rows);                            // hessian matrix
        Vector<vType> direction = Vector<vType>::Zero(rows);    // direction vector (zero when initialized)

        // 设置目标函数的梯度值等
        opt_obj_fn(x_p, grad, hess, opt_data);
    }

    namespace impl {
        template<typename vType>
        bool newton_impl(Vector<vType> &init_vals,
                         std::function<vType(const Vector <vType> &in_vec, Vector <vType> &grad, Matrix <vType> &hess,
                                             void *opt_data)> opt_obj_fn,
                         void *opt_data,
                         settings *set) {

        }
    }
}