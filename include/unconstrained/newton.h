//
// Created by lei on 22-11-1.
//

#ifndef NUMERICOPTIM_NEWTON_H
#define NUMERICOPTIM_NEWTON_H

#include <Eigen/Core>
#include "../utils/optim_settings.h"

/**
 * @brief Newton's Nonlinear Optimization Algorithm
 *
 * @param init_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_obj_fn the function to be minimized, taking three arguments:
 *   - \c in_vec a vector of inputs;
 *   - \c grad a vector to store the gradient;
 *   - \c hess a matrix to store the Hessian;
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

template<typename vType>
using ColVec = Eigen::Matrix<vType, Eigen::Dynamic, 1>;

template<typename vType>
using Mat = Eigen::Matrix<vType, Eigen::Dynamic, Eigen::Dynamic>;

namespace optim {
    template<typename vType>
    bool newton(ColVec<vType> &init_vals,
                std::function<vType(const ColVec<vType> &in_vec, ColVec<vType> &grad, Mat<vType> &hess,
                                    void *opt_data)> opt_obj_fn,
                void *opt_data,
                settings *set);
}

namespace impl {
    template<typename vType>
    bool newton_impl(ColVec<vType> &init_vals,
                     std::function<vType(const ColVec<vType> &in_vec, ColVec<vType> &grad, Mat<vType> &hess,
                                         void *opt_data)> opt_obj_fn,
                     void *opt_data,
                     settings *set);
}

#endif //NUMERICOPTIM_NEWTON_H
