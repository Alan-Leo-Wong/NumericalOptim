//
// Created by lei on 22-11-4.
//

#ifndef NUMERICOPTIM_LBFGSMAT_H
#define NUMERICOPTIM_LBFGSMAT_H

#include "../../utils/geometry.h"

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
        // Constructor
        LBFGSMat() {};

        /**
         * reset internal variables;
         * @param n dimension of the vector to be optimized
         * @param m maximum number of corrections to approximate the Hessian Matrix
         */
        inline void reset(int n, int m) {
            m_m = m;
            m_theta = Scalar(1);
            m_s.resize(n, m);
            m_y.resize(n, m);
            m_ys.resize(m);
            m_alpha.resize(m);
            m_corr_num = 0;
            m_ptr = 0;
        }

        /**
         * fill internal variables;
         * @param s 输入的变化量 x_{n + 1} - x_{n}
         * @param y 梯度的变化量
         */
        inline void add_correction(const RefConstVec<Scalar> &s, const RefConstVec<Scalar> &y) {
            const int loc = m_ptr % m_m; // 指向当前最新加入的用于修正海塞矩阵的向量

//            std::cout << "m_s = " << m_s << std::endl;
            m_s.col(loc).noalias() = s;
//            std::cout << "m_y = " << m_y << std::endl;
            m_y.col(loc).noalias() = y;

            // ys 代表 y^Ts = 1/rho
            const Scalar ys = m_y.col(loc).dot(m_s.col(loc));
            std::cout << "ys = " << ys << std::endl;
            m_ys[loc] = ys;

            m_theta = m_y.col(loc).squaredNorm() / ys;

            if (m_corr_num < m_m) m_corr_num++;

            m_ptr = loc + 1;
        }

        /**
         * Recursive formula to compute the final (descent) direction of LBFGS: inv(H) * desc_grad,
         * where H0 = (1/theta) * I is the initial approximation of the Hessian Matrix H
         * @param grad grad of objective function f at iteration #iter
         * @param a a scalar, its default value is -1 to achieve descent grad direction
         * @param res_dir inv(H) * desc_grad, which is initialized by a * grad
         */
        void apply_Hg(const Vector<Scalar> &grad, const Scalar &a, Vector<Scalar> &res_dir) {
            // Step1: initialize res_dir
            res_dir.resize(grad.size());

            // Step2: compute right product
            res_dir.noalias() = a * grad;
            size_t j = m_ptr % m_m;
            for (size_t i = 0; i < m_corr_num; ++i) {
                // reverse order
                // j: m_m - 1 --> 0
                j = (j + m_m - 1) % m_m;

                // scalar \alpha_j = \rho_j * {s_j}^T * res_dir
                std::cout << "loop1: m_ys[j]: " << m_ys[j] << std::endl;
                m_alpha[j] = m_s.col(j).dot(res_dir) / m_ys[j]; // ys = 1/rho

                // res_dir = res_dir - \alpha_j * y_j
                res_dir.noalias() -= m_alpha[j] * m_y.col(j);
            }

            // Step3: compute center
            // res_dir = inv(H0) * res_dir, H0 = \theta * I
            std::cout << "m_theta " << m_theta << std::endl;
            res_dir /= m_theta;

            // Step4: compute left product
            for (size_t i = 0; i < m_corr_num; ++i) {
                // scalar beta = \rho_j * {y_j}^T * res_dir
                std::cout << "loop2: m_ys[j]: " << m_ys[j] << std::endl;
                const Scalar beta = m_y.col(j).dot(res_dir) / m_ys[j];

                // res_dir = res_dir + (\alpha_j - \beta) * s_j
                res_dir.noalias() += (m_alpha[j] - beta) * m_s.col(j);

                // normal order
                // j: 0 --> m_m - 1
                j = (j + 1) % m_m;
            }
        }
    }; // class end
} // namespace optim

#endif //NUMERICOPTIM_LBFGSMAT_H
