/*
 *  File: kastaun.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.
#pragma once

// Robust primitive variable recovery as described in Kastaun et al. (2020)
// IMPORTANT: The following functions are stolen directly from:
// Phoebus: https://github.com/lanl/phoebus (con2prim_robust.hpp)
// AthenaK: https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenak (ideal_c2p_mhd.hpp)
// They have been lightly adapted to fit into KHARMA,
// and hopefully original authors should be clear from comments

// General template
// We define a specialization based on the Inverter::Type parameter
#include "invert_template.hpp"

#include "coordinate_utils.hpp"
//#include "floors_functions.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"

// This isn't a vecloop, also it takes an argument.
// Left it in since it's useful and all over Phoebus, maybe we'll adopt it
#define SPACELOOP(i) for (int i = 0; i < 3; i++)
#define SPACELOOP2(i, j) SPACELOOP(i) SPACELOOP(j)
#define SPACETIMELOOP(mu) for (int mu = 0; mu < GR_DIM; mu++)

namespace Inverter {


template<typename T, typename Callable>
KOKKOS_FUNCTION
T bisect(Callable&& f, const T& a, const T& b, const T& atol){
    // does not care for number of iterations, just keeps running until atol is met or
    // machine precision is reached. Returns the best guess so far if machine precision is reached.
    T err = 2*atol+1;
    T _a = a;
    T _b = b;
    T c = a;
    T fm;

    assert((f(a) * f(b) <= 0) && "Root not bracketed" );
    
    while (err > atol){
        c = (_a+_b)/2;
        if (c == _a || c == _b){
            // reached machine precision limit, return the best guess so far
            break;
        }
        fm = f(c);
        if (f(_a) * fm  > 0){
            _a = c;
        }
        else{
            _b = c;
        }
        err = std::abs(fm);
    }

    return _b;
}

/**
 * Residual class from Phoebus.
 * Caches function arguments which won't change during solve
 */
class KastaunResidual {
    public:
        KOKKOS_FUNCTION
        KastaunResidual(Real tau, Real d, Real s_sq, Real sb_sq, Real B_sq, Real Gam, Real h0 = 1) : D(d), h0(h0), Gam(Gam) {
            q       = tau / D;
            b_sq    = B_sq / D;
            r_sq    = s_sq / (D * D);
            rb_sq   = sb_sq / (D * D * D);
            z0_sq   = r_sq / (h0 * h0);
            v0_sq   = z0_sq / (1 + z0_sq);
        }

        KOKKOS_FORCEINLINE_FUNCTION
        Real x_mu(const Real mu)
        {
            return 1 / (1 + mu * b_sq);
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real rbar_sq(const Real mu) {
            const Real x = x_mu(mu);
            return x * (x * r_sq + mu * (1.0 + x) * rb_sq);
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real qbar_mu(const Real mu) {
            const Real x = x_mu(mu);
            const Real mux = mu * x;
            return q - 0.5 * (b_sq + mux * mux * (b_sq * r_sq - rb_sq));
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real vsq_hat(const Real mu) {
            Real mu_sq_rbarsq = mu * mu * rbar_sq(mu);
            if (mu_sq_rbarsq < v0_sq){
                return mu_sq_rbarsq;
            } else {
                return v0_sq;
            }
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real W_sq(const Real mu)
        {
            Real vsq_val = mu * mu * rbar_sq(mu);
            if (vsq_val < v0_sq) {
                return 1.0 / (1.0 - vsq_val);
            } else {
                return 1 + z0_sq;
            }
        }

        KOKKOS_FORCEINLINE_FUNCTION
        Real iW_sq(const Real mu)
        {
            Real vsq_val = mu * mu * rbar_sq(mu);
            if (vsq_val < v0_sq) {
                return 1.0 - vsq_val;
            } else {
                return 1.0 / (1.0 + z0_sq);
            }
        }


        KOKKOS_FORCEINLINE_FUNCTION
        Real W(const Real mu)
        {
            return std::sqrt(W_sq(mu));
        }

        KOKKOS_FORCEINLINE_FUNCTION
        Real rho_hat(const Real mu) {
            return D * std::sqrt(iW_sq(mu));
        }
        KOKKOS_FORCEINLINE_FUNCTION
        Real ehat_mu(const Real mu)
        {
            const Real W_val = W(mu);
            const Real q_bar_val = qbar_mu(mu);
            const Real rr_bar_val = rbar_sq(mu);
            const Real wminus1 = W_val - 1.0;
            Real kinetic;
            if (wminus1 < 1e-3) {
                const Real v2_hat_val = vsq_hat(mu);
                kinetic = v2_hat_val * W_val * W_val / (1.0 + W_val);
            } else {
                kinetic = wminus1;
            }
            return W_val * (q_bar_val - mu * rr_bar_val) + kinetic;
        }

        KOKKOS_FORCEINLINE_FUNCTION
        Real ahat_mod(const Real mu)
        {
            const Real Phat_val = std::max(Phat(mu), 0.);
            const Real rhohat_val = std::max(rho_hat(mu), 0.);
            const Real eps_val = std::max(this->ehat_mu(mu), 0.);
            return Phat_val / rhohat_val;
        }

        KOKKOS_FORCEINLINE_FUNCTION
        Real Phat(const Real mu)
        {
            const Real rhohat_val = this->rho_hat(mu);
            const Real ehat_val = ehat_mu(mu);
            return ehat_val * rhohat_val * (Gam - 1.0);
        }

        // Evaluate residual at a value of mu.
        // Kastaun eqn 44
        KOKKOS_INLINE_FUNCTION
        Real obj_fun(const Real mu) {
            const Real x = x_mu(mu);
            const Real rbarsq = rbar_sq(mu);
            const Real qbar = qbar_mu(mu);
            const Real vhatsq = vsq_hat(mu);
            const Real What = W(mu);
            const Real iWhat = std::sqrt(iW_sq(mu));
            const Real rhohat = std::max(rho_hat(mu), 0.);
            const Real ehat = std::max(ehat_mu(mu), 0.);
            // TODO this is ideal-only
            const Real ahat_mod = ehat * (Gam - 1.0);

            // nu_A = h_hat / W = (1 + a_hat)(1 + eps) / W  (paper eq 46)
            const Real nua = (1.0 + ehat + ahat_mod) * iWhat;
            // nu_B = (1 + a_hat) * (1 + qbar - mu*rbarsq)  (paper eq 47)
            const Real nub = (1.0 + ahat_mod / (1 + ehat)) * (1.0 + qbar - mu * rbarsq);
            const Real nuhat = std::max(nua, nub);

            return mu * (nuhat + mu * rbarsq) - 1;
        }

        // Residual for finding bracket values
        // Kastaun eqn 49
        KOKKOS_FORCEINLINE_FUNCTION
        Real bound_obj_fun(const Real mu) {
            Real rbar_val = rbar_sq(mu);
            return mu * mu * (h0*h0 + rbar_val) - 1;
        }

    // private:
    Real D, h0, Gam; //provided
    Real r_sq, rb_sq, b_sq, q, z0_sq, v0_sq; //derived / normalized
};


KOKKOS_INLINE_FUNCTION void get_prims(Real& P, Real& rho, Real& lfac, Real& mu_out, const Real tau, const Real D, const Real s_sq, const Real sb_sq, const Real B_sq, const Real Gam, const Real tol) {
    const Real h0 = 1.;
    KastaunResidual res(tau, D, s_sq, sb_sq, B_sq, Gam, h0);
    // Find upper bound for mu using eqn 49
    Real mu_min = 0;
    Real mu_max;

    if (res.r_sq < h0 * h0) {
        mu_max = 1.0 / h0;
    } else {
        mu_max = bisect([&res](const Real mu){
            return res.bound_obj_fun(mu);
        }, mu_min, 1.0/h0, 0.0);

        // Per Kastaun et al. Sec IV A: nudge mu_max slightly upward to
        // guarantee that the master function root is strictly contained.
        // For extreme inputs (r^2 >> 1), f(mu_+) is theoretically >= 0
        // but floating-point roundoff can make it slightly negative.
        mu_max = std::min(mu_max * (1.0 + 4.0 * tol) + 4.0 * tol, 1.0 / h0);
    }
    
    Real f_upper = res.obj_fun(mu_max);

    if (f_upper <= 0.) {
        // mu_max is already at or past the root (can happen for extreme
        // inputs where ehat < 0 is clamped).  mu_max ~ root, use directly.
        mu_out = mu_max;
    } else {
        mu_out = bisect([&res](const Real mu){
            return res.obj_fun(mu);
        }, mu_min, mu_max, tol);
    }

    P = std::max(res.Phat(mu_out), 0.);
    rho = std::max(res.rho_hat(mu_out), 0.);
    lfac = res.W(mu_out);
}

/**
 * Robust inversion scheme from Kastaun et al. 2020
 * Unholy mashup of the transformation/equations from Phoebus (which are coordinate-general),
 * and the solver from AthenaK (which is easier to read and precomputes the bracket)
 * TODO keep mu between calls to speed up convergence
 * TODO better returns: be explicit about pre- and post-inversion floors, cat neg_input too
 */
template <>
KOKKOS_INLINE_FUNCTION int u_to_p<Type::kastaun>(const GRCoordinates& G, const VariablePack<Real>& U, const VarMap& m_u,
                                              const Real& gam, const int& k, const int& j, const int& i,
                                              const VariablePack<Real>& P, const VarMap& m_p,
                                              const Loci& loc, const int& max_iterations, const Real& tol,
                                              const bool recover_velocity)
{
    // Shouldn't need this, KHARMA should die on NaN
    // But it's here for debugging
    // int num_nans = std::isnan(U(m_u.RHO, k, j, i)) + std::isnan(U(m_u.U1, k, j, i)) + std::isnan(U(m_u.UU, k, j, i));
    // if (num_nans > 0) return static_cast<int>(Status::neg_input);

    // This exists only to keep the math stable on first call,
    // so we can add floors instead of failing outright
    if (U(m_u.RHO, k, j, i) < 1e-20) {
        U(m_u.RHO, k, j, i) = 1e-20;
    }

    // Transform GRMHD variables for the SRMHD Kastaun solver
    const Real alpha  = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));
    const Real a_over_g = alpha / G.gdet(loc, j, i);

    const Real &Urho = U(m_u.RHO, k, j, i);
    const Real D = Urho * a_over_g;

    Real Qcov[GR_DIM] = {(U(m_u.UU, k, j, i) - Urho) * a_over_g,
                    U(m_u.U1, k, j, i) * a_over_g,
                    U(m_u.U2, k, j, i) * a_over_g,
                    U(m_u.U3, k, j, i) * a_over_g};

    const Real ncov[GR_DIM] = {(Real) -alpha, 0., 0., 0.};
    Real ncon[GR_DIM];
    G.raise(ncov, ncon, k, j, i, loc);
    const Real q = (-dot(Qcov, ncon) - D) / D; // TODO floor on this?

    // r_i
    Real rcov[3] = {U(m_u.U1, k, j, i) / Urho,
                    U(m_u.U2, k, j, i) / Urho,
                    U(m_u.U3, k, j, i) / Urho};
    Real rcon[3];
    Real gupper[GR_DIM][GR_DIM];
    G.gcon(loc, j, i, gupper);
    // Ripped from AthenaK's "TransformToSRMHD,"
    // since we don't use the spatial metric anywhere else.  Original comment:
    // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
    //       g^0i = beta^i/alpha^2
    //       g^00 = -1/ alpha^2
    // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
    rcon[0] = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*rcov[0] +
                (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*rcov[1] +
                (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*rcov[2]);  // (C26)

    rcon[1] = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*rcov[0] +
                (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*rcov[1] +
                (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*rcov[2]);  // (C26)

    rcon[2] = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*rcov[0] +
                (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*rcov[1] +
                (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*rcov[2]);  // (C26)

    Real rsq = 0.0;
    SPACELOOP(ii) rsq += rcon[ii]*rcov[ii];

    Real bsq = 0.0;
    Real bsq_rpsq = 0.0;
    Real rbsq = 0.0;
    Real bdotr = 0.0;
    Real bu[] = {0.0, 0.0, 0.0};
    // If the magnetic field is being evolved at all...
    if (m_u.B1 >= 0) {
        const Real sD = 1.0 / m::sqrt(D);
        // b^i
        SPACELOOP(ii) {
            bu[ii] = (U(m_u.B1 + ii, k, j, i) * a_over_g) * sD;
            bdotr += bu[ii] * rcov[ii];
        }
        SPACELOOP2(ii, jj) bsq += G.gcov(loc, j, i, ii + 1, jj + 1) * bu[ii] * bu[jj];
        bsq = std::max(0.0, bsq);

        rbsq = bdotr * bdotr;
        bsq_rpsq = bsq * rsq - rbsq;
    }
    // Compute non-normalized quantities for get_prims
    const Real tau = -dot(Qcov, ncon) - D;  // tau/D = q
    const Real s_sq = rsq * D * D;
    const Real sb_sq = rbsq * D * D * D;
    const Real B_sq = bsq * D;

    // Solve using get_prims
    Real P_prim, rho_prim, W, mu;
    get_prims(P_prim, rho_prim, W, mu, tau, D, s_sq, sb_sq, B_sq, gam, tol);

    // Set primitive variables
    // These values should be as *raw* as possible, whether or not they respect the floors
    // (or even physics).  We will add material and try again if they're bad
    P(m_p.RHO, k, j, i) = std::max(rho_prim, 0.);
    P(m_p.UU, k, j, i) = std::max(P_prim / (gam - 1.0), 0.);  // u = P / (gam - 1) for ideal gas

    // Set velocities
    // TODO make sure W*mu*x really should be >0
    // Latter part is a vector/signed quantity, don't set a minimum at 0
    const Real x = 1.0 / (1.0 + mu * bsq);
    SPACELOOP(ii) P(m_p.U1 + ii, k, j, i) = std::max(W * mu * x, 0.) * (rcon[ii] + mu * bdotr * bu[ii]);

    // If we should try to recover velocity, do it in this function
    if (!recover_velocity) {
        // bisect always converges, so just return success
        return static_cast<int>(Status::success);
    } else {
        // Calculate P->U on the inverted values
        const Real rho = P(m_p.RHO, k, j, i);
        const Real u = P(m_p.UU, k, j, i);
        const Real uvec[NVEC] = {P(m_p.U1, k, j, i), P(m_p.U2, k, j, i), P(m_p.U3, k, j, i)};
        const Real B_P[NVEC] = {P(m_p.B1, k, j, i), P(m_p.B2, k, j, i), P(m_p.B3, k, j, i)};
        Real rho_ut = 0., T[GR_DIM] = {0.};
        GRMHD::p_to_u_mhd(G, rho, u, uvec, B_P, gam, k, j, i, rho_ut, T);
        const Real bad_vel_tolerance = 200 * tol;
        // If we didn't conserve momentum within a loose tolerance based on the solver tol...
        if ((std::abs((T[1] - U(m_u.U1, k, j, i)) / U(m_u.U1, k, j, i)) > bad_vel_tolerance) ||
            (std::abs((T[2] - U(m_u.U2, k, j, i)) / U(m_u.U2, k, j, i)) > bad_vel_tolerance) ||
            (std::abs((T[3] - U(m_u.U3, k, j, i)) / U(m_u.U3, k, j, i)) > bad_vel_tolerance)) {

            // ...then solve so function ehat(mu) Kastaun matches the existing value,
            // and use that to reset the velocities.
            // This prioritizes the new thermal energy in the total T^0_0,
            // but dumps any extra back into the velocities to be less disruptive.
            // TODO either set this on better theory or redo the algebra and condense it

            // Create residual object for velocity recovery
            KastaunResidual res(tau, D, s_sq, sb_sq, B_sq, gam);

            const Real e_actual = P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);
            auto f = [&res, e_actual] (Real mu_val) {
                return res.ehat_mu(mu_val) - e_actual;
            };

            // Rootfind for mu that would have produced the current e
            bool e_solve_failed = false;
            Real mu_new = mu, mum = 0., mup = 1.;
            if (f(mum) * f(mup) > 0.) {
                e_solve_failed = true;
            } else {
                while (true) {
                    Real muc = (mum + mup) / 2.;
                    Real resv = m::abs(f(muc));
                    if (resv < 1e-8 || m::abs((mup - mum) / 2) < 1e-10) {
                        mu_new = muc;
                        e_solve_failed = (resv > 1e-8);
                        break;
                    }
                    // Same sign as left side -> center now left side
                    if (f(muc) * f(mum) > 0.)
                        mum = muc;
                    else
                        mup = muc;
                }
            }

            // Reset only velocities with new mu
            const Real x_new = 1.0 / (1.0 + mu_new * bsq);
            const Real W_new = res.W(mu_new);
            SPACELOOP(ii) P(m_p.U1 + ii, k, j, i) = std::max(W_new * mu_new * x_new, 0.) * (rcon[ii] + mu_new * bdotr * bu[ii]);
            return (e_solve_failed) ? static_cast<int>(Status::bad_velocity) : static_cast<int>(Status::floor);
        } else {
            return static_cast<int>(Status::success);
        }
    }
}

} // namespace Inverter
