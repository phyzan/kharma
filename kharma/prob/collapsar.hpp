#pragma once

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "decs.hpp"
#include "internal/csv_reader.hpp"
#include "internal/csv_row.hpp"
#include "types.hpp"
#include "floors.hpp"
#include <stdexcept>
#include <csv.hpp>

using namespace parthenon;

namespace kharma_constants {

constexpr Real MSUN = 1.989e33;    // Solar mass in grams
constexpr Real GNEWT = 6.674e-8;   // G in cm³/g/s²
constexpr Real CL = 2.998e10;      // c in cm/s
    
} // namespace constants

class DeviceInterpolator1D {

public:

    DeviceInterpolator1D(const std::vector<Real>& x, const std::vector<Real>& y) : x_("x interp", x.size()), y_("y interp", y.size()), n(x.size()) {
        if (x.size() < 2){
            PARTHENON_THROW("x and y must have at least two elements");
        } else if (x.size() != y.size()) {
            PARTHENON_THROW("x and y must have the same size");
        }
        auto h_x = Kokkos::create_mirror_view(x_);
        auto h_y = Kokkos::create_mirror_view(y_);
        for (size_t i=0; i<x.size(); ++i) {
            h_x(i) = x[i];
            h_y(i) = y[i];
        }
        Kokkos::deep_copy(x_, h_x);
        Kokkos::deep_copy(y_, h_y);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    Real operator()(Real x) const {
        if (x <= x_(0)){
            return y_(0);
        } else if (x >= x_(n - 1)){
            return y_(n - 1);
        }
        size_t idx = find_left_idx(x);
        Real x0 = x_[idx];
        Real x1 = x_[idx + 1];
        Real y0 = y_[idx];
        Real y1 = y_[idx + 1];

        // Linear interpolation formula
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    const m::View<Real*>& get_x() const { return x_; }
    KOKKOS_FORCEINLINE_FUNCTION
    const m::View<Real*>& get_y() const { return y_; }
    KOKKOS_FORCEINLINE_FUNCTION
    size_t size() const { return n; }

private:

    KOKKOS_FORCEINLINE_FUNCTION
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

    m::View<Real*> x_;
    m::View<Real*> y_;
    size_t n;
};


inline TaskStatus InitializeCollapsar(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput* pin){

    MeshBlock* pmb = rc->GetBlockPointer();
    //Get block-specific parameters
    const std::string file = pin->GetString("collapsar", "profile_file");
    const Real M_BH = pin->GetReal("collapsar", "M_BH"); // in solar masses
    const Real density_unit = pin->GetReal("collapsar", "density_unit"); // in g/cm³

    const Real M_SUN = kharma_constants::MSUN;    // Solar mass in grams
    const Real G = kharma_constants::GNEWT;   // G in cm³/g/s²
    const Real c = kharma_constants::CL;      // light speed in cm/s
    const Real time_unit = G * M_BH * M_SUN / (c * c * c); // in seconds
    const Real length_unit = G * M_BH * M_SUN / (c * c); // in cm
    const Real pressure_unit = density_unit * c * c; // in erg/cm³

    
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const Coordinates_t& GR = pmb->coords;
    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");


    //Read the file with numerical values
    if (file.empty()) {
        PARTHENON_THROW("No profile file specified for collapsar problem");
    }


    // Get the data of each column in the file
    std::vector<Real> r, rho_r, Omega_r, u_r;
    csv::CSVReader reader(file);
    for (csv::CSVRow& row : reader) {
        Real r_val = row["r"].get<Real>() / length_unit; // Convert to code units
        Real rho_val = row["rho_r"].get<Real>() / density_unit;
        Real Omega_val = row["Omega_r"].get<Real>() * time_unit; // Convert to code units
        Real P_val = row["P_r"].get<Real>() / pressure_unit;
        r.push_back(r_val);
        rho_r.push_back(rho_val);
        Omega_r.push_back(Omega_val);
        u_r.push_back(P_val/(gam - 1.0));
    }
    const Real r_max = r.back();
    const Real r_min = r.front();
    std::cout << "rmin = " << r_min << std::endl;
    std::cout << "rmax = " << r_max << std::endl;
    std::cout << "length unit = " << length_unit << std::endl;
    
    DeviceInterpolator1D rho_interp(r, rho_r);
    DeviceInterpolator1D Omega_interp(r, Omega_r);
    DeviceInterpolator1D u_interp(r, u_r);

    // Define unit quantities to convert to code units


    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("collapsar_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (int k, int j, int i) {
            Real X[GR_DIM];
            GR.coord_embed(k, j, i, Loci::center, X);
            const GReal r = X[1];
            if (r <= r_max && r >= r_min) {

                Real Omega = Omega_interp(r);
                Real g_tt = GR.gcov(Loci::center, j, i, 0, 0);
                Real g_tphi = GR.gcov(Loci::center, j, i, 0, 3);
                Real g_phiphi = GR.gcov(Loci::center, j, i, 3, 3);
                Real ut = 1.0/m::sqrt(-g_tt - 2.0*g_tphi*Omega - g_phiphi*Omega*Omega);

                Real ucon[GR_DIM] = {ut, 0.0, 0.0, Omega*ut};
                Real gcon[GR_DIM][GR_DIM];
                Real u_prim[NVEC];
                GR.gcon(Loci::center, j, i, gcon);
                fourvel_to_prim(gcon, ucon, u_prim);

                rho(k,j,i) = rho_interp(r);
                u(k,j,i) = u_interp(r);
                uvec(0, k, j, i) = u_prim[0];
                uvec(1, k, j, i) = u_prim[1];
                uvec(2, k, j, i) = u_prim[2];
            }
        });

    // set physical scales
    pin->SetReal("collapsar", "units.length", length_unit);
    pin->SetReal("collapsar", "units.time", time_unit);
    pin->SetReal("collapsar", "units.density", density_unit);
    pin->SetReal("collapsar", "units.pressure", pressure_unit);
    pin->SetReal("collapsar", "units.magnetic_field", c*m::sqrt(4.0*M_PI*density_unit));

    Floors::ApplyInitialFloors(pin, rc.get(), IndexDomain::interior);
    return TaskStatus::complete;
}