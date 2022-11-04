//
// Created by lei on 22-11-4.
//

#ifndef NUMERICOPTIM_BFGS_H
#define NUMERICOPTIM_BFGS_H

#include "../utils/geometry.h"
#include "../utils/optim_settings.h"
#include "lbfgsMat.h"

namespace optim {
    using namespace geometry;

    template<typename Scalar>
    class LBFGSSolver {
    private:
        const settings::LBFGSParam<Scalar> &m_param;    // Parameters to control the LBFGS algorithm
        LBFGSMat<Scalar> m_H;                           // Approximation to the Hessian matrix
        Vector<Scalar> m_fx;                            // History of the objective function values
        Vector<Scalar> m_px;                            // Old(past) x
        Vector<Scalar> m_grad;                          // Current gradient
        Scalar m_gnorm;                                 // Norm of the current gradient
        Vector<Scalar> m_pgrad;                         // Old(past) gradient
        Vector<Scalar> m_dir;                           // Moving(Searching) direction

    public:
        // Reset internal variables
        // n: dimension of the vector to be optimized
        inline void reset(int n);

        LBFGSSolver(const settings::LBFGSParam<Scalar> &param) : m_param(param) {
            m_param.check_param();
        }

        /// Minimizing a multivariate function using the L-BFGS algorithm.
        /// Exceptions will be thrown if error occurs.
        ///
        /// \param f  A function object such that `f(x, grad)` returns the
        ///           objective function value at `x`, and overwrites `grad` with
        ///           the gradient.
        /// \param x  In: An initial guess of the optimal point.
        ///           Out: The best point found.
        /// \param fx Out: The objective function value at `x`.
        ///
        /// \return Number of iterations used.
        ///
        template<typename Func>
        inline int minimize(Func &f, Vector<Scalar> x, Scalar &fx) {
            using std::abs;

            // Dimension of the vector
            const size_t n = x.size();

            // The length of iteration lag for objective function value to test convergence
            const int f_past = m_param.past;

            // Evaluate function and compute the current gradient
            fx = f(x, m_grad);
            m_gnorm = m_grad.norm();
            if (f_past > 0) m_fx[0] = fx;

            // Early exit if the initial x is already the best minimizer
            if (m_gnorm <= m_param.epsilon || m_gnorm <= m_param.epsilon * x.norm()) {
                return 1;
            }

            // Initial direction (negative gradient)
            m_dir.noalias() = -m_grad;
            // Initial step size (\alpha)
            Scalar step = Scalar(1) / m_dir.norm();

            // Number of iterations used
            int k = 1;

            while (true) {
                // Save the current x and gradient
                m_px.noalias() = x;
                m_pgrad.noalias() = m_grad;

                // compute grad.dot(dir), i.e. grad^T * dir
                Scalar gd = m_grad.dot(m_dir);

                const Scalar step_max = m_param.ls.max_step;


            }
        }
    };
}

#endif //NUMERICOPTIM_BFGS_H
