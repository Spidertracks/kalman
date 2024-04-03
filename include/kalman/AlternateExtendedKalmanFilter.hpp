// The MIT License (MIT)
//
// Copyright (c) 2015 Markus Herb
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef KALMAN_ALTEXTENDEDKALMANFILTER_HPP_
#define KALMAN_ALTEXTENDEDKALMANFILTER_HPP_

#include "KalmanFilterBase.hpp"
#include "StandardFilterBase.hpp"
#include "LinearizedSystemModel.hpp"
#include "LinearizedMeasurementModel.hpp"

namespace Kalman
{

    /**
     * @brief Extended Kalman Filter (EKF)
     *
     * This implementation is based upon [An Introduction to the Kalman Filter](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
     * by Greg Welch and Gary Bishop.
     *
     * @param StateType The vector-type of the system state (usually some type derived from Kalman::Vector)
     */
    template <class StateType>
    class AltExtendedKalmanFilter : public KalmanFilterBase<StateType>,
                                    public StandardFilterBase<StateType>
    {
    public:
        //! Kalman Filter base type
        typedef KalmanFilterBase<StateType> KalmanBase;
        //! Standard Filter base type
        typedef StandardFilterBase<StateType> StandardBase;

        //! Numeric Scalar Type inherited from base
        using typename KalmanBase::T;

        //! State Type inherited from base
        using typename KalmanBase::State;

        //! Linearized Measurement Model Type
        template <class Measurement, template <class> class CovarianceBase>
        using MeasurementModelType = LinearizedMeasurementModel<State, Measurement, CovarianceBase>;

        //! Linearized System Model Type
        template <class Control, template <class> class CovarianceBase>
        using SystemModelType = LinearizedSystemModel<State, Control, CovarianceBase>;

    protected:
        //! Kalman Gain Matrix Type
        template <class Measurement>
        using KalmanGain = Kalman::KalmanGain<State, Measurement>;

    protected:
        //! State Estimate
        using KalmanBase::x;
        //! State Covariance Matrix
        using StandardBase::P;

    public:
        /**
         * @brief Constructor
         */
        AltExtendedKalmanFilter()
        {
            // Setup state and covariance
            P.setIdentity();
        }

        /**
         * @brief Perform filter prediction step using system model and no control input (i.e. \f$ u = 0 \f$)
         *
         * @param [in] s The System model
         * @param [in] f A function that is called after the state is updated. Eg to renormalise a quaternion
         * @return The updated state estimate
         */
        template <class Control, template <class> class CovarianceBase>
        const State &predict(SystemModelType<Control, CovarianceBase> &s, std::function<void(State &)> f = nullptr)
        {
            // predict state (without control)
            Control u;
            u.setZero();
            return predict(s, u, f);
        }

        /**
         * @brief Perform filter prediction step using control input \f$u\f$ and corresponding system model
         *
         * @param [in] s The System model
         * @param [in] u The Control input vector
         * @param [in] f A function that is called after the state is updated. Eg to renormalise a quaternion
         * @return The updated state estimate
         */
        template <class Control, template <class> class CovarianceBase>
        const State &predict(SystemModelType<Control, CovarianceBase> &s, const Control &u, std::function<void(State &)> f = nullptr)
        {
            s.updateJacobians(x, u);

            // predict state
            x = s.f(x, u);
            if (f)
            {
                f(x);
            }

            // predict covariance
            P = (s.F * P * s.F.transpose()) + s.Q;

            // return state prediction
            return this->getState();
        }

        /**
         * @brief Perform filter update step using measurement \f$z\f$ and corresponding measurement model
         *
         * @param [in] m The Measurement model
         * @param [in] z The measurement vector
         * @param [in] f A function that is called after the state is updated. Eg to renormalise a quaternion
         * @return The updated state estimate
         */
        template <class Measurement, template <class> class CovarianceBase>
        const State &update(MeasurementModelType<Measurement, CovarianceBase> &m, const Measurement &z, std::function<void(State&)> f = nullptr)
        {
            // More numerically stable implementation of the EKF update
            // P = (I-KH)P(I-KH)' + KRK'

            const Measurement expected_observation = m.h( x );
            m.updateJacobians( x );

            const KalmanGain<Measurement> pht = P * m.H.transpose();
            const Covariance<Measurement> expected_observation_covariance = m.H * pht;
            const Measurement residuals = z - expected_observation;
            const Covariance<Measurement> residuals_covariance = expected_observation_covariance + m.R;

            KalmanGain<Measurement> K = pht * residuals_covariance.inverse();
            x += K * residuals;
            if (f){
                f(x);
            }

            const Covariance<State> I_kh = Covariance<State>::Identity() - K * m.H;
            const Covariance<State> sensor_noise_covariance = K * m.R * K.transpose();
            P = I_kh * P * I_kh.transpose() + sensor_noise_covariance;

            // return updated state estimate
            return this->getState();
        }
    };
}

#endif
