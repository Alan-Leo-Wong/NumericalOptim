//
// Created by lei on 22-11-4.
//
#include "../../include/line_search/LineSearchMoreThuente.h"

namespace optim {
    template<typename Scalar>
    template<typename Func, typename SolverParam>
    void LineSearchMoreThuente<Scalar>::LineSearch(Func &f, const SolverParam &param, const Vector<Scalar> px,
                                                   const Vector<Scalar> dir, const Scalar &step_max, Scalar &step,
                                                   Scalar &fx, Vector<Scalar> &grad, Scalar &dg, Vector<Scalar> x) {
        using std::abs;

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
        const Scalar dg_init = dg;
        // std::cout << "fx_init = " << fx_init << ", dg_init = " << dg_init << std::endl << std::endl;

        // Make sure dir is a descent direction
        // 'dg_init >= Scalar(0)' means the absolute angle between dir and gradient is less than 90 degrees
        if (dg_init >= Scalar(0))
            throw std::logic_error("the moving direction does not decrease the objective function value");

        // Tolerance for convergence test
        // Sufficient decrease(Armijo)
        const Scalar test_dec = param.ftol * dg_init;
        // Curvature condition
        const Scalar test_curv = -param.wolfe * dg_init;

        // Initialized the bracketing interval
        Scalar I_lo = Scalar(0), I_hi = std::numeric_limits<Scalar>::infinity();
        Scalar fI_lo = Scalar(0), fI_hi = std::numeric_limits<Scalar>::infinity();
        Scalar gI_lo = (Scalar(1) - param.ftol) * dg_init, gI_hi = std::numeric_limits<Scalar>::infinity();

        // Function value and gradient at the current step size
        x.noalias() = px + step * dir; // update to new point by using the initial direction and step
        fx = f(x, grad); // Gradient of the new point x
        dg = grad.dot(dir); // Projection of gradient on the search direction of the new point x


    }
}