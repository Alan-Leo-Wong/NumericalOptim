//
// Created by lei on 22-11-4.
//

#include "../../include/unconstrained/lbfgsMat.h"

namespace optim {
    template<typename Scalar>
    void LBFGSMat<Scalar>::reset(int n, int m) {
        m_m = m;
        m_theta = Scalar(1);
        m_s.resize(n, m);
        m_y.resize(n, m);
        m_alpha.resize(m);
        m_corr_num = 0;
        m_ptr = 0;
    }

    template<typename Scalar>
    void LBFGSMat<Scalar>::add_correction(const RefConstVec<Scalar> &s, const RefConstVec<Scalar> &y) {
        const int loc = m_ptr % m_m; // 指向当前最新加入的用于修正海塞矩阵的向量

        m_s.col(loc).noalias() = s;
        m_s.col(loc).noalias() = y;

        // ys 代表 y^Ts = 1/rho
        const Scalar ys = m_y.col(loc).dot(m_s.col(loc));
        m_ys[loc] = ys;

        m_theta = m_y.col(loc).squaredNorm() / ys;

        if (m_corr_num < m_m) m_corr_num++;

        m_ptr = loc + 1;
    }

    template<typename Scalar>
    void LBFGSMat<Scalar>::apply_Hg(const Vector<Scalar> grad, const Scalar &a, Vector<Scalar> &res_dir) {
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
            m_alpha[j] = m_s.col(j).dot(res_dir) / m_ys[j]; // ys = 1/rho

            // res_dir = res_dir - \alpha_j * y_j
            res_dir.noalias() -= m_alpha[j] * m_y.col(j);
        }

        // Step3: compute center
        // res_dir = inv(H0) * res_dir, H0 = \theta * I
        res_dir /= m_theta;

        // Step4: compute left product
        for (size_t i = 0; i < m_corr_num; ++i) {
            // scalar beta = \rho_j * {y_j}^T * res_dir
            const Scalar beta = m_y.dot(res_dir) / m_ys[j];

            // res_dir = res_dir + (\alpha_j - \beta) * s_j
            res_dir += (m_alpha[j] - beta) * m_s[j];

            // normal order
            // j: 0 --> m_m - 1
            j = (j + 1) % m_m;
        }
    }
}
