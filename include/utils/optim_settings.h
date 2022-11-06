//
// Created by lei on 22-11-1.
//

#ifndef NUMERICOPTIM_OPTIM_SETTINGS_H
#define NUMERICOPTIM_OPTIM_SETTINGS_H

namespace optim {
    namespace settings {
        enum LINE_SEARCH_TERMINATION_CONDITION {
            ///
            /// Backtracking method with the Armijo condition.
            /// The backtracking method finds the step length such that it satisfies
            /// the sufficient decrease (Armijo) condition,
            /// \f$f(x + a \cdot d) \le f(x) + \beta' \cdot a \cdot g(x)^T d\f$,
            /// where \f$x\f$ is the current point, \f$d\f$ is the current search direction,
            /// \f$a\f$ is the step length, and \f$\beta'\f$ is the value specified by
            /// \ref LBFGSParam::ftol. \f$f\f$ and \f$g\f$ are the function
            /// and gradient values respectively.
            ///
            LINESEARCH_BACKTRACKING_ARMIJO = 1,

            ///
            /// The backtracking method with the defualt (regular Wolfe) condition.
            /// An alias of `LINESEARCH_BACKTRACKING_WOLFE`.
            ///
            LINESEARCH_BACKTRACKING = 2,

            ///
            /// Backtracking method with regular Wolfe condition.
            /// The backtracking method finds the step length such that it satisfies
            /// both the Armijo condition (`LINESEARCH_BACKTRACKING_ARMIJO`)
            /// and the curvature condition,
            /// \f$g(x + a \cdot d)^T d \ge \beta \cdot g(x)^T d\f$, where \f$\beta\f$
            /// is the value specified by \ref LBFGSParam::wolfe.
            ///
            LINESEARCH_BACKTRACKING_WOLFE = 2,

            ///
            /// Backtracking method with strong Wolfe condition.
            /// The backtracking method finds the step length such that it satisfies
            /// both the Armijo condition (`LINESEARCH_BACKTRACKING_ARMIJO`)
            /// and the following condition,
            /// \f$\vert g(x + a \cdot d)^T d\vert \le \beta \cdot \vert g(x)^T d\vert\f$,
            /// where \f$\beta\f$ is the value specified by \ref LineSearch::wolfe.
            ///
            LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3
        };

        template<typename Scalar = double>
        struct LineSearch {
            ///
            /// The line search termination condition.
            /// This parameter specifies the line search termination condition that will be used.
            /// The default value is `LINESEARCH_BACKTRACKING_STRONG_WOLFE`.
            ///
            int line_search_type = LINESEARCH_BACKTRACKING_STRONG_WOLFE;

            ///
            /// The maximum number of trials for the line search.
            /// This parameter controls the number of function and gradients evaluations
            /// per iteration for the line search routine. The default value is \c 20.
            ///
            int max_line_search = 20;

            ///
            /// The minimum step length allowed in the line search.
            /// The default value is \c 1e-20. Usually this value does not need to be
            /// modified.
            ///
            Scalar min_step = Scalar(1e-20);

            ///
            /// The maximum step length allowed in the line search.
            /// The default value is \c 1e+20. Usually this value does not need to be
            /// modified.
            ///
            Scalar max_step = Scalar(1e+20);

            ///
            /// A parameter to control the accuracy of the line search routine.
            /// The default value is \c 1e-4. This parameter should be greater
            /// than zero and smaller than \c 0.5.
            ///
            Scalar ftol = Scalar(1e-4);

            ///
            /// The coefficient for the Wolfe condition.
            /// This parameter is valid only when the line-search
            /// algorithm is used with the Wolfe condition.
            /// The default value is \c 0.9. This parameter should be greater
            /// the \ref ftol parameter and smaller than \c 1.0.
            ///
            Scalar wolfe = Scalar(0.9);

            LineSearch() {}

            LineSearch(int ls_type, int ls_iter, Scalar min_step, Scalar max_step, Scalar ftol, Scalar wolfe) :
                    line_search_type(ls_type), max_line_search(ls_iter), min_step(min_step),
                    max_step(max_step), ftol(ftol), wolfe(wolfe) {}
        };

        template<typename Scalar = double>
        struct GDParam {
            ///
            /// Absolute tolerance for convergence test.
            /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
            /// with which the solution is to be found. A minimization terminates when
            /// \f$||g|| < \max\{\epsilon_{abs}, {rel}_\epsilon||x||\}\f$,
            /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
            /// \c 1e-5.
            ///
            Scalar epsilon;

            ///
            /// Relative tolerance for convergence test.
            /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
            /// with which the solution is to be found. A minimization terminates when
            /// \f$||g|| < \max\{\epsilon_{abs}, {rel}_\epsilon||x||\}\f$,
            /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
            /// \c 1e-5.
            ///
            Scalar rel_epsilon;

            ///
            /// The maximum number of iterations.
            /// The optimization process is terminated when the iteration count
            /// exceeds this parameter. Setting this parameter to zero continues an
            /// optimization process until a convergence or error. The default value
            /// is \c 0.
            ///
            int max_iterations;

            ///
            /// Distance for delta-based convergence test.
            /// This parameter determines the distance \f$d\f$ to compute the
            /// rate of decrease of the objective function,
            /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
            /// step. If the value of this parameter is zero, the delta-based convergence
            /// test will not be performed. The default value is \c 0.
            ///
            int past;

            ///
            /// Delta for convergence test.
            /// The algorithm stops when the following condition is met,
            /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
            /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
            /// \f$d\f$ iterations ago (specified by the \ref past parameter).
            /// The default value is \c 0.
            ///
            Scalar delta;

            ///
            /// Whether use line search algorithm
            ///
            bool use_ls = true;

            ///
            /// The parameters about line search algorithm.
            ///
            LineSearch<Scalar> ls = LineSearch<Scalar>();

            ///
            /// Checking the validity of Gradient Descent parameters.
            /// An `std::invalid_argument` exception will be thrown if some parameter
            /// is invalid.
            ///
            inline void check_param() const {
                if (epsilon < 0)
                    throw std::invalid_argument("'epsilon' must be non-negative");
                if (rel_epsilon < 0)
                    throw std::invalid_argument("'epsilon_rel' must be non-negative");
                if (past < 0)
                    throw std::invalid_argument("'past' must be non-negative");
                if (delta < 0)
                    throw std::invalid_argument("'delta' must be non-negative");
                if (max_iterations < 0)
                    throw std::invalid_argument("'max_iterations' must be non-negative");
                if (ls.line_search_type < LINESEARCH_BACKTRACKING_ARMIJO ||
                    ls.line_search_type > LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    throw std::invalid_argument("unsupported line search termination condition");
                if (ls.max_line_search <= 0)
                    throw std::invalid_argument("'max_linesearch' must be positive");
                if (ls.min_step < 0)
                    throw std::invalid_argument("'min_step' must be positive");
                if (ls.max_step < ls.min_step)
                    throw std::invalid_argument("'max_step' must be greater than 'min_step'");
                if (ls.ftol <= 0 || ls.ftol >= 0.5)
                    throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
                if (ls.wolfe <= ls.ftol || ls.wolfe >= 1)
                    throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
            }

            GDParam() {}

            GDParam(Scalar ep, Scalar rel_ep, int max_iter, int p, Scalar del, bool use, LineSearch<Scalar> lineSearch)
                    : epsilon(ep), rel_epsilon(rel_ep), max_iterations(max_iter), past(p), delta(del), use_ls(use) {
                if (use_ls) ls = lineSearch;
            }
        };

        ///TODO
        template<typename Scalar = double>
        struct NewtonParam {
            ///
            /// Absolute tolerance for convergence test.
            /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
            /// with which the solution is to be found. A minimization terminates when
            /// \f$||g|| < \max\{\epsilon_{abs}, {rel}_\epsilon||x||\}\f$,
            /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
            /// \c 1e-5.
            ///
            Scalar epsilon;

            ///
            /// Relative tolerance for convergence test.
            /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
            /// with which the solution is to be found. A minimization terminates when
            /// \f$||g|| < \max\{\epsilon_{abs}, {rel}_\epsilon||x||\}\f$,
            /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
            /// \c 1e-5.
            ///
            Scalar rel_epsilon;

            ///
            /// Distance for delta-based convergence test.
            /// This parameter determines the distance \f$d\f$ to compute the
            /// rate of decrease of the objective function,
            /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
            /// step. If the value of this parameter is zero, the delta-based convergence
            /// test will not be performed. The default value is \c 0.
            ///
            int past;

            ///
            /// Delta for convergence test.
            /// The algorithm stops when the following condition is met,
            /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
            /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
            /// \f$d\f$ iterations ago (specified by the \ref past parameter).
            /// The default value is \c 0.
            ///
            Scalar delta;

            ///
            /// The maximum number of iterations.
            /// The optimization process is terminated when the iteration count
            /// exceeds this parameter. Setting this parameter to zero continues an
            /// optimization process until a convergence or error. The default value
            /// is \c 0.
            ///
            int max_iterations;

            ///
            /// Whether use line search algorithm
            ///
            bool use_ls = true;

            ///
            /// The parameters about line search algorithm.
            ///
            LineSearch<Scalar> ls = LineSearch<Scalar>();

            ///
            /// Checking the validity of Newton parameters.
            /// An `std::invalid_argument` exception will be thrown if some parameter
            /// is invalid.
            ///
            inline void check_param() const {
                if (epsilon < 0)
                    throw std::invalid_argument("'epsilon' must be non-negative");
                if (rel_epsilon < 0)
                    throw std::invalid_argument("'epsilon_rel' must be non-negative");
                if (past < 0)
                    throw std::invalid_argument("'past' must be non-negative");
                if (delta < 0)
                    throw std::invalid_argument("'delta' must be non-negative");
                if (max_iterations < 0)
                    throw std::invalid_argument("'max_iterations' must be non-negative");
                if (ls.line_search_type < LINESEARCH_BACKTRACKING_ARMIJO ||
                    ls.line_search_type > LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    throw std::invalid_argument("unsupported line search termination condition");
                if (ls.max_line_search <= 0)
                    throw std::invalid_argument("'max_linesearch' must be positive");
                if (ls.min_step < 0)
                    throw std::invalid_argument("'min_step' must be positive");
                if (ls.max_step < ls.min_step)
                    throw std::invalid_argument("'max_step' must be greater than 'min_step'");
                if (ls.ftol <= 0 || ls.ftol >= 0.5)
                    throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
                if (ls.wolfe <= ls.ftol || ls.wolfe >= 1)
                    throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
            }
        };

        template<typename Scalar = double>
        struct LBFGSParam {
            ///
            /// The number of corrections to approximate the inverse Hessian matrix.
            /// The L-BFGS routine stores the computation results of previous \ref m
            /// iterations to approximate the inverse Hessian matrix of the current
            /// iteration. This parameter controls the size of the limited memories
            /// (corrections). The default value is \c 6. Values less than \c 3 are
            /// not recommended. Large values will result in excessive computing time.
            ///
            int m = 6;

            ///
            /// Absolute tolerance for convergence test.
            /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
            /// with which the solution is to be found. A minimization terminates when
            /// \f$||g|| < \max\{\epsilon_{abs}, {rel}_\epsilon||x||\}\f$,
            /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
            /// \c 1e-5.
            ///
            Scalar epsilon = Scalar(1e-5);

            ///
            /// Relative tolerance for convergence test.
            /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
            /// with which the solution is to be found. A minimization terminates when
            /// \f$||g|| < \max\{\epsilon_{abs}, {rel}_\epsilon||x||\}\f$,
            /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
            /// \c 1e-5.
            ///
            Scalar rel_epsilon = Scalar(1e-5);

            ///
            /// Distance for delta-based convergence test.
            /// This parameter determines the distance \f$d\f$ to compute the
            /// rate of decrease of the objective function,
            /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
            /// step and \f$d\f$ is this parameter.
            /// If the value of this parameter is zero, the delta-based convergence
            /// test will not be performed. The default value is \c 0.
            ///
            int past = 0;

            ///
            /// Delta for convergence test.
            /// The algorithm stops when the following condition is met,
            /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
            /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
            /// \f$d\f$ iterations ago (specified by the \ref past parameter).
            /// The default value is \c 0.
            ///
            Scalar delta = Scalar(0);

            ///
            /// The maximum number of iterations.
            /// The optimization process is terminated when the iteration count
            /// exceeds this parameter. Setting this parameter to zero continues an
            /// optimization process until a convergence or error. The default value
            /// is \c 10.
            ///
            int max_iterations = 10;

            ///
            /// Whether use line search algorithm
            ///
            bool use_ls = true;

            ///
            /// The parameters about line search algorithm.
            ///
            LineSearch<Scalar> ls;

            ///
            /// Checking the validity of L-BFGS parameters.
            /// An `std::invalid_argument` exception will be thrown if some parameter
            /// is invalid.
            ///
            inline void check_param() const {
                if (m <= 0)
                    throw std::invalid_argument("'m' must be positive");
                if (epsilon < 0)
                    throw std::invalid_argument("'epsilon' must be non-negative");
                if (rel_epsilon < 0)
                    throw std::invalid_argument("'epsilon_rel' must be non-negative");
                if (past < 0)
                    throw std::invalid_argument("'past' must be non-negative");
                if (delta < 0)
                    throw std::invalid_argument("'delta' must be non-negative");
                if (max_iterations < 0)
                    throw std::invalid_argument("'max_iterations' must be non-negative");
                if (ls.line_search_type < LINESEARCH_BACKTRACKING_ARMIJO ||
                    ls.line_search_type > LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    throw std::invalid_argument("unsupported line search termination condition");
                if (ls.max_line_search <= 0)
                    throw std::invalid_argument("'max_linesearch' must be positive");
                if (ls.min_step < 0)
                    throw std::invalid_argument("'min_step' must be positive");
                if (ls.max_step < ls.min_step)
                    throw std::invalid_argument("'max_step' must be greater than 'min_step'");
                if (ls.ftol <= 0 || ls.ftol >= 0.5)
                    throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
                if (ls.wolfe <= ls.ftol || ls.wolfe >= 1)
                    throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
            }

            LBFGSParam() {}

            LBFGSParam(int m, Scalar ep, Scalar rel_ep, int max_iter, int p, Scalar del, bool use,
                       LineSearch<Scalar> lineSearch)
                    : m(m), epsilon(ep), rel_epsilon(rel_ep), max_iterations(max_iter), past(p), delta(del),
                      use_ls(use) {
                if (use_ls) ls = lineSearch;
            }
        };

        // for all above algorithm settings
        /*template<typename Scalar = double, class AlParam = LBFGSParam<Scalar>, typename ... params>
        struct ParamSetting {
            AlParam alParam = AlParam(params);
            void check_param() const {

            }
        };*/
    }
}

#endif //NUMERICOPTIM_OPTIM_SETTINGS_H
