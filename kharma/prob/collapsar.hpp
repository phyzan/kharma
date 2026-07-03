#pragma once

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "decs.hpp"
#include "internal/csv_reader.hpp"
#include "internal/csv_row.hpp"
#include "types.hpp"
#include <stdexcept>
#include <csv.hpp>

using namespace parthenon;




class Interpolator1D {

public:

    Interpolator1D(const std::vector<Real>& x, const std::vector<Real>& y) : x_(x), y_(y) {}

    Real operator()(Real x) const {
        if (x <= x_.front()){
            return y_.front();
        } else if (x >= x_.back()){
            return y_.back();
        }
        size_t idx = find_left_idx(x);
        Real x0 = x_[idx];
        Real x1 = x_[idx + 1];
        Real y0 = y_[idx];
        Real y1 = y_[idx + 1];

        // Linear interpolation formula
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }

private:

    size_t find_left_idx(Real x) const {
        // use binary search to find the index of the left value
        size_t left = 0;
        size_t right = x_.size() - 1;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (x_[mid] < x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left - 1;
    }

    std::vector<Real> x_;
    std::vector<Real> y_;
};


TaskStatus InitializeCollapsar(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin){

    // Make sure automatic seeding is disabled, since the magnetic field is given in the profile file
    if (pin->GetString("b_field", "type") != "none") {
        PARTHENON_THROW("Automatic seeding must be disabled for the collapsar problem");
    }

    MeshBlock* pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const Coordinates_t& G = pmb->coords;
    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");


    //Read the file with numerical values
    std::string file = pin->GetOrAddString("collapsar", "profile_file", "");
    if (file.empty()) {
        PARTHENON_THROW("No profile file specified for collapsar problem");
    }


    // Get the data of each column in the file
    std::vector<Real> r, rho_r, P_r, v_phi, u_r;
    csv::CSVReader reader(file);
    for (csv::CSVRow& row : reader) {
        r.push_back(row["r"].get<Real>());
        Real rho = row["rho_r"].get<Real>();
        Real P = row["P_r"].get<Real>();
        rho_r.push_back(rho);
        v_phi.push_back(row["v_phi"].get<Real>());
        P_r.push_back(P);
        u_r.push_back(P/((gam - 1.0) * rho));
    }


    PARTHENON_THROW("Collapsar problem is not yet implemented. Please use the explosion problem instead.");




    // IndexDomain domain = IndexDomain::interior;
    // IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    // IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    // IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    // pmb->par_for("collapsar_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    //     KOKKOS_LAMBDA (int k, int j, int i) {
    //         Real X[GR_DIM];
    //         G.coord_embed(k, j, i, Loci::center, X);
    //         const GReal r = m::sqrt(X[1]*X[1] + X[2]*X[2] + X[3]*X[3]);

    //         // Set up a simple collapsar profile
    //         rho(k,j,i) = 1.0e-4 * m::exp(-r*r);
    //         u(k,j,i) = 1.0e-5 / (gam - 1.0) * m::exp(-r*r);
    //         uvec(1,k,j,i) = 0.0;
    //         uvec(2,k,j,i) = 0.0;
    //         uvec(3,k,j,i) = 0.0;
    //     });


    return TaskStatus::complete;
}