//
// Created by lei on 22-11-4.
//

#ifndef NUMERICOPTIM_BFGS_H
#define NUMERICOPTIM_BFGS_H

#include "../../utils/geometry.h"
#include "../../utils/optim_settings.h"
#include "lbfgsMat.h"
#include "../../line_search/LineSearchMoreThuente.h"

namespace optim {
    using namespace geometry;

    template<typename Scalar,
            template<class> class LineSearch = LineSearchMoreThuente>
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
        inline void reset(int n) {
            const int m = m_param.m;
            m_H.reset(n, m);
            m_px.resize(n);
            m_grad.resize(n);
            m_pgrad.resize(n);
            m_dir.resize(n);
            if (m_param.past > 0)
                m_fx.resize(m_param.past);
        }

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
        inline int minimize(Func &f, Vector<Scalar> &x, Scalar &fx) {
            using std::abs;

            // Dimension of the vector
            const size_t n = x.size();
            reset(n);

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
            int iter = 1;

            while (true) {
                // Save the current x and gradient
                m_px.noalias() = x;
                m_pgrad.noalias() = m_grad;
                // compute grad.dot(dir), i.e. grad^T * dir
                Scalar gd = m_grad.dot(m_dir);
                const Scalar step_max = m_param.ls.max_step;

                /// Line Search for step
                // Line search to update x, fx and gradient
                LineSearch<Scalar>::LineSearch(f, m_param, m_px, m_dir,
                                               step_max, step,
                                               fx, m_grad, gd, x);
                // New gradient norm
                m_gnorm = m_grad.norm();

                // std::cout << "Iter " << iter << " finished line search" << std::endl;
                // std::cout << "   x = " << x.transpose() << std::endl;
                // std::cout << "   f(x) = " << fx << ", ||grad|| = " << m_gnorm << std::endl << std::endl;

                /// Convergence test
                // Convergence test -- gradient
                if (m_gnorm <= m_param.epsilon || m_gnorm <= m_param.rel_epsilon * x.norm()) {
                    return iter;
                }
                // Convergence test -- objective function value
                if (f_past > 0) {
                    const Scalar fx_d = m_fx[iter % f_past];
                    if (iter >= f_past &&
                        abs(fx_d - fx) <= m_param.delta *
                                          std::max(std::max(abs(fx), abs(fx_d)), Scalar(1)))
                        return iter;
                    m_fx[iter % f_past] = fx;
                }
                // Maximum number of iterations
                if (m_param.max_iterations != 0 && iter >= m_param.max_iterations)
                    return iter;

                /// Approximate Hessian Matrix H to find direction
                // Update s and y
                // Let k = iter
                // s_{k+1} = x_{k+1} - x_k
                // y_{k+1} = g_{k+1} - g_k
                m_H.add_correction(x - m_px, m_grad - m_pgrad);

                // Recursive formula to compute m_dir = -inv(m_H) * g (g = m_grad)
                m_H.apply_Hg(m_grad, -Scalar(1), m_dir);

                /// Ready to next iteration
                // Reset step = 1.0 as initial guess for the next line search
                step = Scalar(1);
                iter++;
            }
        }

        ///
        /// Returning the gradient vector on the last iterate.
        /// Typically used to debug and test convergence.
        /// Should only be called after the `minimize()` function.
        ///
        /// \return A const reference to the gradient vector.
        ///
        const Vector<Scalar> &final_grad() const { return m_grad; }

        ///
        /// Returning the Euclidean norm of the final gradient.
        ///
        Scalar final_grad_norm() const { return m_gnorm; }
    }; // class end
} // namespace optim

#endif //NUMERICOPTIM_BFGS_H
