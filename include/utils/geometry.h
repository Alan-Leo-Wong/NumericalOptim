//
// Created by lei on 22-11-4.
//

#ifndef NUMERICOPTIM_GEOMETRY_H
#define NUMERICOPTIM_GEOMETRY_H

#include <Eigen/Core>

namespace optim {
    namespace geometry {
        /*typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vectord;
        typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vectorf;
        typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Vectori;
        typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> Vectorui;

        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrixd;
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrixf;
        typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Matrixi;
        typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> Matrixui;*/
        template<typename Scalar>
        using Vector =  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        template<typename Scalar>
        using Matrix =  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        template<typename Scalar>
        using RefVec = Eigen::Ref<Vector<Scalar>>;

        template<typename Scalar>
        using RefConstVec = Eigen::Ref<const Vector<Scalar>>;
    }
}

#endif //NUMERICOPTIM_GEOMETRY_H
