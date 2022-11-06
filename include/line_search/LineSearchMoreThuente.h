//
// Created by lei on 22-11-4.
//

#ifndef NUMERICOPTIM_LINESEARCHMORETHUENTE_H
#define NUMERICOPTIM_LINESEARCHMORETHUENTE_H

#include <stdexcept>    // std::invalid_argument, std::runtime_error
#include <Eigen/Core>
#include "../utils/geometry.h"
#include "../utils/optim_settings.h"

///
/// The line search algorithm by Moré and Thuente (1994), currently used for the L-BFGS-B algorithm.
///
/// The target of this line search algorithm is to find a step size \f$\alpha\f$ that satisfies the strong Wolfe condition
/// \f$f(x+\alpha d) \le f(x) + \alpha\mu g(x)^T d\f$ and \f$|g(x+\alpha d)^T d| \le \eta|g(x)^T d|\f$.
/// Our implementation is a simplified version of the algorithm in [1]. We assume that \f$0<\mu<\eta<1\f$, while in [1]
/// they do not assume \f$\eta>\mu\f$. As a result, the algorithm in [1] has two stages, but in our implementation we
/// only need the first stage to guarantee the convergence.
///
/// Reference:
/// [1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease.
///

namespace optim {
    using namespace geometry;

    template<typename Scalar = double>
    class LineSearchMoreThuente {
    private:
        /// The following minimizer will be used to trial step selection.

        // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
        // that interpolates fa, ga, and fb, assuming the minimizer exists
        // For case I: fb >= fa and ga * (b - a) < 0
        // a and b (maybe a > b or b > a) are endpoints of the interval I;
        // fa and fb are the value of f at the point a and b;
        // ga and gb are the value of f'(gradient of f) at the point a and b;
        static Scalar quadratic_minimizer(const Scalar &a, const Scalar &b,
                                          const Scalar &fa, const Scalar &fb,
                                          const Scalar &ga) {
            const Scalar ba = b - a;
            const Scalar w = Scalar(0.5) * ba * ga / (fa - fb + ba * ga);
            return a + w * ba;
        }

        // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
        // that interpolates fa, ga and gb, assuming the minimizer exists
        // The result actually does not depend on fa
        // For case II: ga * (b - a) < 0, ga * gb < 0
        // For case III: ga * (b - a) < 0, ga * ga >= 0, |gb| <= |ga|
        static Scalar quadratic_minimizer(const Scalar &a, const Scalar &b,
                                          const Scalar &ga, const Scalar &gb) {
            const Scalar ba = b - a;
            const Scalar w = ga / (ga - gb);
            return a + w * ba;
        }

        // Local minimizer of a cubic function q(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
        // that interpolates fa, ga, fb and gb, assuming a != b
        // Also sets a flag indicating whether the minimizer exists
        static Scalar cubic_minimizer(const Scalar &a, const Scalar &b,
                                      const Scalar &fa, const Scalar &fb,
                                      const Scalar &ga, const Scalar &gb, bool &exists) {
            using std::abs;
            using std::sqrt;

            const Scalar apb = a + b;
            const Scalar ba = b - a;
            const Scalar ba2 = ba * ba;
            const Scalar fba = fb - fa;
            const Scalar gba = gb - ga;
            // z3 = c3 * (b-a)^3, z2 = c2 * (b-a)^3, z1 = c1 * (b-a)^3
            const Scalar z3 = (ga + gb) * ba - Scalar(2) * fba;
            const Scalar z2 = Scalar(0.5) * (gba * ba2 - Scalar(3) * apb * z3);
            const Scalar z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;
            // std::cout << "z1 = " << z1 << ", z2 = " << z2 << ", z3 = " << z3 << std::endl;

            // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
            const Scalar eps = std::numeric_limits<Scalar>::epsilon();
            if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1)) {
                // Minimizer exists if c2 > 0
                exists = (z2 * ba > Scalar(0));
                // Return the end point if the minimizer does not exist
                return exists ? (-Scalar(0.5) * z1 / z2) : b;
            }

            // Now we can assume z3 > 0
            // The minimizer is a solution to the equation c1 + 2*c2 * x + 3*c3 * x^2 = 0
            // roots = -(z2/z3) / 3 (+-) sqrt((z2/z3)^2 - 3 * (z1/z3)) / 3
            //
            // Let u = z2/(3z3) and v = z1/z2
            // The minimizer exists if v/u <= 1
            const Scalar u = z2 / (Scalar(3) * z3), v = z1 / z2;
            const Scalar vu = v / u;
            exists = (vu <= Scalar(1));
            if (!exists)
                return b;

            // We need to find a numerically stable way to compute the roots, as z3 may still be small
            //
            // If |u| >= |v|, let w = 1 + sqrt(1-v/u), and then
            // r1 = -u * w, r2 = -v / w, r1 does not need to be the smaller one
            //
            // If |u| < |v|, we must have uv <= 0, and then
            // r = -u (+-) sqrt(delta), where
            // sqrt(delta) = sqrt(|u|) * sqrt(|v|) * sqrt(1-u/v)
            Scalar r1 = Scalar(0), r2 = Scalar(0);
            if (abs(u) >= abs(v)) {
                const Scalar w = Scalar(1) + sqrt(Scalar(1) - vu);
                r1 = -u * w;
                r2 = -v / w;
            } else {
                const Scalar sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1 - u / v);
                r1 = -u - sqrtd;
                r2 = -u + sqrtd;
            }
            return (z3 * ba > Scalar(0)) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
        }

        /// Select the next step size according to the current step sizes(Section 4, Page 298 of the paper),
        // Interval values a, function values f, and derivatives values g (f')
        // _l and _u are the endpoints of the interval I
        static Scalar step_selection(
                const Scalar &al, const Scalar &au, const Scalar &at,
                const Scalar &fl, const Scalar &fu, const Scalar &ft,
                const Scalar &gl, const Scalar &gu, const Scalar &gt) {
            using std::abs;

            if (al == au)
                return al;

            // If ft = Inf or gt = Inf, we return the middle point of al and at
            if (!std::isfinite(ft) || !std::isfinite(gt))
                return (al + at) / Scalar(2);

            // Case 1: ft > fl
            // ac: cubic interpolation of fl, ft, gl, gt
            // aq: quadratic interpolation of fl, gl, ft
            bool ac_exists;
            // std::cout << "al = " << al << ", at = " << at << ", fl = " << fl << ", ft = " << ft << ", gl = " << gl << ", gt = " << gt << std::endl;
            const Scalar ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
            const Scalar aq = quadratic_minimizer(al, at, fl, gl, ft);
            // std::cout << "ac = " << ac << ", aq = " << aq << std::endl;
            if (ft > fl) {
                // This should not happen if ft > fl, but just to be safe
                if (!ac_exists)
                    return aq;
                // Then use the scheme described in the paper
                return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / Scalar(2));
            }

            // Case 2: ft <= fl, gt * gl < 0
            // as: quadratic interpolation of gl and gt
            const Scalar as = quadratic_minimizer(al, at, gl, gt);
            if (gt * gl < Scalar(0))
                return (abs(ac - at) >= abs(as - at)) ? ac : as;

            // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
            const Scalar deltal = Scalar(1.1), deltau = Scalar(0.66);
            if (abs(gt) < abs(gl)) {
                // We choose either ac or as
                // The case for ac: 1. It exists, and
                //                  2. ac is farther than at from al, and
                //                  3. ac is closer to at than as
                // Cases for as: otherwise
                const Scalar res = (ac_exists &&
                                    (ac - at) * (at - al) > Scalar(0) &&
                                    abs(ac - at) < abs(as - at)) ?
                                   ac :
                                   as;
                // Postprocessing the chosen step
                return (at > al) ?
                       std::min(at + deltau * (au - at), res) :
                       std::max(at + deltau * (au - at), res);
            }

            // Simple extrapolation if au, fu, or gu is infinity
            if ((!std::isfinite(au)) || (!std::isfinite(fu)) || (!std::isfinite(gu)))
                return at + deltal * (at - al);

            // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
            // ae: cubic interpolation of ft, fu, gt, gu
            bool ae_exists;
            const Scalar ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
            // The following is not used in the paper, but it seems to be a reasonable safeguard
            return (at > al) ?
                   std::min(at + deltau * (au - at), ae) :
                   std::max(at + deltau * (au - at), ae);
        }

    public:
        ///
        /// Line search by Moré and Thuente (1994).
        ///
        /// \param f        A function object such that `f(x, grad)` returns the
        ///                 objective function value at `x`, and overwrites `grad` with
        ///                 the gradient.
        /// \param param    For example, `LBFGSParam` object that stores the
        ///                 parameters of corresponding solver.
        /// \param px       Store the current point and to calculate new point x. So it also
        ///                 represents old(past) x.
        /// \param dir      The current moving direction.
        /// \param step_max The upper bound for the step size that makes x feasible.
        /// \param step     In: The initial step length.
        ///                 Out: The calculated satisfied step length.
        /// \param fx       In: The objective function value at the current point.
        ///                 Out: The function value at the new point.
        /// \param grad     In: The current gradient vector.
        ///                 Out: The gradient at the new point.
        /// \param gd       In: The inner product between dir and grad.
        ///                 Out: The inner product between dir and the new gradient.
        /// \param x        Out: The new point moved to. x = px + dir * step
        ///
        template<typename Func, typename SolverParam>
        static void LineSearch(Func &f, const SolverParam &param,
                               const Vector<Scalar> px, const Vector<Scalar> dir,
                               const Scalar &step_max, Scalar &step,
                               Scalar &fx, Vector<Scalar> &grad,
                               Scalar &gd, Vector<Scalar>& x) {

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
            Scalar g_psiI_lo =
                    (Scalar(1) - param.ls.ftol) * gd_init, g_psiI_hi = std::numeric_limits<Scalar>::infinity();

            // Function value and gradient at the current step size
            x = px + step * dir; // update to new point by using the initial direction and step
            // grad = f'(x + step * dir)
            fx = f(x, grad); // Value of the new point x
            // gd = f'(x + step * dir) * dir
            gd = grad.dot(dir); // Projection of gradient on the search direction of the new point x

            // Convergence test
            // Maybe initial value already have converged
            if (fx <= fx_init + step * test_dec && std::abs(gd) <= test_curv) {
                /*std::cout << "** Criteria met\n\n";
                std::cout << "========================= Leaving line search =========================\n\n";*/
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

                    // std::cout << "Case 1: new step = " << new_step << std::endl;
                } else if (g_psi_t * (I_lo - step) > Scalar(0)) {
                    /// Case 2: psi_t <= psi_l and g_psi_t * (al -at) > 0
                    // Page 291 of Moré and Thuente (1994) suggests that
                    // new_at = min(at + delta * (at - al), step_max), delta in [1.1, 4]
                    new_step = std::min(step + delta * (step - I_lo), step_max);

                    I_lo = step;
                    psiI_lo = psi_t;
                    g_psiI_lo = g_psi_t;

                    // std::cout << "Case 2: new step = " << new_step << std::endl;
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

                    // std::cout << "Case 3: new step = " << new_step << std::endl;
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
                    /*std::cout << "** Maximum step size reached\n\n";
                    std::cout << "========================= Leaving line search =========================\n\n";*/
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

                // std::cout << "step = " << step << ", fx = " << fx << ", gd = " << gd << std::endl;

                // Convergence test
                if (fx <= fx_init + step * test_dec && std::abs(gd) <= test_curv) {
                    /*std::cout << "** Criteria met\n\n";
                    std::cout << "========================= Leaving line search =========================\n\n";*/
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
                        /*std::cout << "** Maximum step size reached\n\n";
                        std::cout << "========================= Leaving line search =========================\n\n";*/
                        return;
                    }
                }
            }

            if (iter >= param.ls.max_line_search)
                throw ::std::runtime_error(
                        "The line search routine reached the maximum number of iterations and no satisfied step was found!");

        } // function LineSearch end
    }; // class end
} // namespace optim

#endif //NUMERICOPTIM_LINESEARCHMORETHUENTE_H
