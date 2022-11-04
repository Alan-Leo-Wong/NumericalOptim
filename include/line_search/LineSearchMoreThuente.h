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

        // Select the next step size according to the current step sizes(Section 4, Page 13 of the paper),
        // Interval values a, function values f, and derivatives g(f')
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
        /// \param dg       In: The inner product between dir and grad.
        ///                 Out: The inner product between dir and the new gradient.
        /// \param x        Out: The new point moved to.
        ///
        template<typename Func, typename SolverParam>
        void LineSearch(Func &f, const SolverParam &param,
                        const Vector<Scalar> px, const Vector<Scalar> dir,
                        const Scalar &step_max, Scalar &step,
                        Scalar &fx, Vector<Scalar> &grad,
                        Scalar &dg, Vector<Scalar> x);
    };
}

#endif //NUMERICOPTIM_LINESEARCHMORETHUENTE_H
