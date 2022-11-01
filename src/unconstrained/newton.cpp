//
// Created by lei on 22-11-1.
//

#include "../../include/unconstrained/newton.h"

namespace optim {
    template<typename vType>
    bool newton(ColVec<vType> &init_vals,
                std::function<vType(const ColVec<vType> &in_vec, ColVec<vType> &grad, Mat<vType> &hess,
                                    void *opt_data)> opt_obj_fn,
                void *opt_data,
                settings *set) {
        bool success = false;

        const size_t rows = init_vals.rows();

        const size_t max_iters = set->max_iters;
        const double grad_err = set->grad_err;

        ColVec<vType> x = init_vals;
        ColVec<vType> x_p = x;                                  // 用于更新 x 的临时值
        ColVec<vType> grad(rows);                               // gradient vector
        Mat<vType> hess(rows, rows);                            // hessian matrix
        ColVec<vType> direction = ColVec<vType>::Zero(rows);    // direction vector (zero when initialized)

        // 目标函数的一些properties设置
        opt_obj_fn(x_p, grad, hess, opt_data);
    }
}

namespace impl {
    template<typename vType>
    bool newton_impl(ColVec<vType> &init_vals,
                     std::function<vType(const ColVec<vType> &in_vec, ColVec<vType> &grad, Mat<vType> &hess,
                                         void *opt_data)> opt_obj_fn,
                     void *opt_data,
                     settings *set) {

    }
}
