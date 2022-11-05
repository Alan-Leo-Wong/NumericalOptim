//
// Created by lei on 22-11-1.
//

#ifndef NUMERICOPTIM_NEWTON_H
#define NUMERICOPTIM_NEWTON_H

#include "../../utils/geometry.h"
#include "../../utils/optim_settings.h"

namespace optim {
    using namespace geometry;

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
    template<typename Scalar>
    bool newton(Vector<Scalar> &init_vals,
                std::function<Scalar(const Vector<Scalar> &in_vec, Vector<Scalar> &grad, Matrix<Scalar> &hess,
                                     void *opt_data)> opt_obj_fn,
                void *opt_data,
                settings::NewtonParam<> *set);

    namespace impl {
        template<typename Scalar>
        bool newton_impl(Vector<Scalar> &init_vals,
                         std::function<Scalar(const Vector<Scalar> &in_vec, Vector<Scalar> &grad, Matrix<Scalar> &hess,
                                              void *opt_data)> opt_obj_fn,
                         void *opt_data,
                         settings::NewtonParam<Scalar> *set);
    }
}

#endif //NUMERICOPTIM_NEWTON_H
