//
// Created by lei on 22-11-1.
//

#include <iostream>
#include "../include/optim.h"

using Eigen::VectorXd;

double obj_func(const VectorXd &x, VectorXd &grad) {
    const int n = x.size(); // dimension

    VectorXd d(n);
    for (int i = 0; i < n; i++)
        d[i] = i;

    double fx = (x - d).squaredNorm(); // (x - d)^T * (x - d), the minimizer is d

    grad.noalias() = 2.0 * (x - d);

    return fx;
}

int main(int argc, char **argv) {
    const int n = 3;

    optim::settings::LBFGSParam<double> lbfgsParam;
    optim::LBFGSSolver<double> lbfgsSolver(lbfgsParam);

    VectorXd x = VectorXd::Ones(n);
    double fx;
    int iter = lbfgsSolver.minimize(obj_func, x, fx);

    std::cout << "------finished------\n";
    std::cout << "passed " << iter << " iterations" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
}