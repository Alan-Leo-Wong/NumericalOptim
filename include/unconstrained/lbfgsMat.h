//
// Created by lei on 22-11-4.
//

#ifndef NUMERICOPTIM_LBFGSMAT_H
#define NUMERICOPTIM_LBFGSMAT_H

#include "../utils/geometry.h"

namespace optim {
    using namespace geometry;

    template<typename Scalar>
    class LBFGSMat {
    private:
        int m_m;                    // Maximum number of correction vectors (只取最近的m个s、y)
        Scalar m_theta;             // theta * I is the initial approximation to the Hessian matrix
        Matrix<Scalar> m_s;         // History of the s vectors
        Matrix<Scalar> m_y;         // History of the y vectors
        Vector<Scalar> m_ys;        // History of the s'y values
        Vector<Scalar> m_alpha;     // Temporary values used in computing H * v
        int m_corr_num;                // Number of correction vectors in the history, m_ncorr <= m
        int m_ptr;                  // A Pointer to locate the most recent history, 1 <= m_ptr <= m
        // Details: s and y vectors are stored in cyclic order.
        //          For example, if the current s-vector is stored in m_s[, m-1],
        //          then in the next iteration m_s[, 0] will be overwritten.
        //          m_s[, m_ptr-1] points to the most recent history,
        //          and m_s[, m_ptr % m] points to the most distant one.

    public:
        LBFGSMat() = default;

        /**
         * reset internal variables;
         * @param n dimension of the vector to be optimized
         * @param m maximum number of corrections to approximate the Hessian Matrix
         */
        void reset(int n, int m);

        /**
         * fill internal variables;
         * @param s 输入的变化量 x_{n + 1} - x_{n}
         * @param y 梯度的变化量
         */
        void add_correction(const RefConstVec<Scalar> &s, const RefConstVec<Scalar> &y);

        /**
         * Recursive formula to compute the descend dir of LBFGS: inv(H) * desc_grad,
         * where H0 = (1/theta) * I is the initial approximation of the Hessian Matrix H
         * @param grad grad
         * @param a a scalar, its default value is -1 to achieve descend grad direction
         * @param res_dir inv(H) * desc_grad, which is initialized by a * grad
         */
        void apply_Hg(const Vector<Scalar> desc_grad, const Scalar &a, Vector<Scalar> &res_dir);
    };
}

#endif //NUMERICOPTIM_LBFGSMAT_H
