//
// Created by lei on 22-11-4.
//
#include "../../include/line_search/LineSearchMoreThuente.h"

namespace optim {
    template<typename Scalar>
    template<typename Func, typename SolverParam>
    void LineSearchMoreThuente<Scalar>::LineSearch(Func &f, const SolverParam &param, const Vector<Scalar> px,
                                                   const Vector<Scalar> dir, const Scalar &step_max, Scalar &step,
                                                   Scalar &fx, Vector<Scalar> &grad, Scalar &gd, Vector<Scalar> x) {

        // std::cout << "========================= Entering line search =========================\n\n";

        /// Initial
        // Check the value of step (\alpha),
        // it must be guaranteed to be greater than 0 and not exceed the maximum value step_max
        if (step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");
        if (step > step_max)
            throw std::invalid_argument("'step' must be less than or equal to 'step_max'");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction (dir.dot(grad))
        const Scalar gd_init = gd; // gd = psi'(0) = f'(x) * d
        // std::cout << "fx_init = " << fx_init << ", gd_init = " << gd_init << std::endl << std::endl;

        // Make sure dir is a descent direction
        // 'gd_init >= Scalar(0)' means the absolute angle between dir and gradient is less than 90 degrees
        if (gd_init >= Scalar(0))
            throw std::logic_error("the moving direction does not decrease the objective function value");

        // Tolerance for convergence test
        // Sufficient decrease(Armijo), param.ls.ftol = mu
        const Scalar test_dec = param.ls.ftol * gd_init; // mu * f'(x) * dir * step
        // Curvature condition
        const Scalar test_curv = -param.ls.wolfe * gd_init;

        // Initialized the bracketing interval
        Scalar I_lo = Scalar(0), I_hi = std::numeric_limits<Scalar>::infinity();
        Scalar psiI_lo = Scalar(0), psiI_hi = std::numeric_limits<Scalar>::infinity();
        Scalar g_psiI_lo = (Scalar(1) - param.ls.ftol) * gd_init, g_psiI_hi = std::numeric_limits<Scalar>::infinity();

        // Function value and gradient at the current step size
        x.noalias() = px + step * dir; // update to new point by using the initial direction and step
        // grad = f'(x + step * dir)
        fx = f(x, grad); // Gradient of the new point x
        // gd = f'(x + step * dir) * dir
        gd = grad.dot(dir); // Projection of gradient on the search direction of the new point x

        // Convergence test
        // Maybe initial value already have converged
        if (fx <= fx_init + step * test_dec && std::abs(gd) <= test_curv) {
            // std::cout << "** Criteria met\n\n";
            // std::cout << "========================= Leaving line search =========================\n\n";
            return;
        }

        ///The Search Algorithm(Section 2, Page 291 from the paper)
        const Scalar delta = Scalar(1.1); // Page 291, Formula 2.2
        int iter = 0;
        for (iter; iter < param.ls.max_line_search; ++iter) {
            // Let step = step_t (trial value)
            // psi_t = psi(step) = f(px + step * dir) - f(px) -  step * mu * f'(px) * dir = fx - fx_init - step * test_dec
            // g_psi_t = psi'(step) = f'(px + step * dir) * dir - mu * f'(px) * dir = gd - param.ls.ftol * gd_init
            const Scalar psi_t = fx - fx_init - step * test_dec;
            const Scalar g_psi_t = gd - param.ls.ftol * gd_init;

            // Update step size and bracketing interval I
            Scalar new_step;
            if (psi_t > psiI_lo) {
                /// Case 1: psi_t > psi_l
                new_step = step_selection(I_lo, I_hi, step,
                                          psiI_lo, psiI_hi, psi_t,
                                          g_psiI_lo, g_psiI_hi, g_psi_t);
                // Sanity check: if the computed new_step is too small,
                // typically due to extremely large value of ft,
                // switch to the middle point
                if (new_step <= param.ls.min_step)
                    new_step = (I_lo + step) / Scalar(2);

                I_hi = step;
                psiI_hi = psi_t;
                g_psiI_hi = g_psi_t;
            } else if (g_psi_t * (I_lo - step) > Scalar(0)) {
                /// Case 2: psi_t <= psi_l and g_psi_t * (al -at) > 0
                // Page 291 of Moré and Thuente (1994) suggests that
                // new_at = min(at + delta * (at - al), step_max), delta in [1.1, 4]
                new_step = std::min(step + delta * (step - I_lo), step_max);

                I_lo = step;
                psiI_lo = psi_t;
                g_psiI_lo = g_psi_t;
            } else {
                /// Case 3: psi_t <= psi_l and g_psi_t * (al -at) <= 0
                new_step = step_selection(I_lo, I_hi, step,
                                          psiI_lo, psiI_hi, psi_t,
                                          g_psiI_lo, g_psiI_hi, g_psi_t);

                I_hi = I_lo;
                psiI_hi = psiI_lo;
                g_psiI_hi = g_psiI_hi;

                I_lo = step;
                psiI_lo = psi_t;
                g_psiI_lo = g_psi_t;
            }

            // Case 1 and 3 are interpolations, whereas Case 2 is extrapolation
            // This means that Case 2 may return new_step = step_max,
            // and we need to decide whether to accept this value
            // 1. If both step and new_step equal to step_max, it means
            //    step will have no further change, so we accept it
            // 2. Otherwise, we need to test the function value and gradient
            //    on step_max, and decide later

            /// In case step, new_step, and step_max are equal,
            // directly return the computed x and fx
            if (step == step_max && new_step >= step_max) {
                // std::cout << "** Maximum step size reached\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";
                return;
            }

            /// Otherwise, recompute x and fx based on new_step
            step = new_step;

            // Check if the new_step satisfies [min_step, max_step]
            if (step < param.ls.min_step)
                throw std::runtime_error("The line search step became smaller than the minimum value allowed");
            if (step > param.ls.max_step)
                throw std::runtime_error("The line search step became larger than the maximum value allowed");

            // Update parameter, function value, and gradient
            x.noalias() = px + step * dir;
            fx = f(x, grad);
            gd = grad.dot(dir);

            // Convergence test
            if (fx <= fx_init + step * test_dec && std::abs(gd) <= test_curv) {
                // std::cout << "** Criteria met\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";
                return;
            }

            // Now assume step = step_max, and we need to decide whether to
            // exit the line search (see the comments above regarding step_max)
            // If we reach here, it means this step size does not pass the convergence test,
            // so either the sufficient decrease condition or the curvature
            // condition is not met yet
            //
            // Typically the curvature condition is harder to meet, and it is
            // possible that no step size in [0, step_max] satisfies the condition
            //
            // But we need to make sure that its psi function value 'is smaller than
            // the best one so far'. If not, go to the next iteration and find a better one
            if (step >= step_max) {
                const Scalar psit = fx - fx_init - step * test_dec;
                if (psit <= psiI_lo) {
                    // std::cout << "** Maximum step size reached\n\n";
                    // std::cout << "========================= Leaving line search =========================\n\n";
                    return;
                }
            }
        }

        if(iter >= param.ls.max_line_search)
            throw::std::runtime_error("The line search routine reached the maximum number of iterations and no satisfied step was found!");
    } // function end
} // namespace optim