#pragma once

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "decs.hpp"
#include "internal/common.hpp"
#include "internal/csv_reader.hpp"
#include "internal/csv_row.hpp"
#include "types.hpp"
#include "floors.hpp"
#include "utils/constants.hpp"
#include <stdexcept>
#include <csv.hpp>

using namespace parthenon;


template<typename T, typename Callable>
T bisect(Callable&& f, const T& lower, const T& upper, const T& atol){
    T err = 2*atol+1;
    T a = lower;
    T b = upper;
    T c = a;
    T fm;

    assert((f(a) * f(b) <= 0) && "Root not bracketed" );
    
    while (err > atol){
        c = (a+b)/2;
        if (c == a || c == b){
            // reached machine precision limit, return the best guess so far
            break;
        }
        fm = f(c);
        if (f(a) * fm  > 0){
            a = c;
        }
        else{
            b = c;
        }
        err = abs<T>(fm);
    }

    return b;
}

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

    const Real M_SUN = MSUN_cgs<Real>;    // Solar mass in grams
    const Real G = GNEWT_cgs<Real>;   // G in cm³/g/s²
    const Real c = CL_cgs<Real>;      // light speed in cm/s
    const Real mass_unit = M_BH * M_SUN; // in grams
    const Real time_unit = G * mass_unit / (c * c * c); // in seconds
    const Real length_unit = G * mass_unit / (c * c); // in cm
    const Real pressure_unit = density_unit * c * c; // in erg/cm³


    // set physical scales
    pin->SetReal("collapsar", "units.length", length_unit);
    pin->SetReal("collapsar", "units.time", time_unit);
    pin->SetReal("collapsar", "units.density", density_unit);
    pin->SetReal("collapsar", "units.pressure", pressure_unit);
    pin->SetReal("collapsar", "units.magnetic_field", c*m::sqrt(4.0*M_PI*density_unit));

    
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
    std::vector<Real> r, rho_r, Omega_r, u_r, M_r;
    csv::CSVReader reader(file);
    for (csv::CSVRow& row : reader) {
        Real r_val = row["r"].get<Real>() / length_unit; // Convert to code units
        Real rho_val = row["rho_r"].get<Real>() / density_unit;
        Real Omega_val = row["Omega_r"].get<Real>() * time_unit; // Convert to code units
        Real P_val = row["P_r"].get<Real>() / pressure_unit;
        Real M_val = row["M_r"].get<Real>() / mass_unit; // Convert to code units
        r.push_back(r_val);
        rho_r.push_back(rho_val);
        Omega_r.push_back(Omega_val);
        u_r.push_back(P_val/(gam - 1.0));
        M_r.push_back(M_val);
    }

    if (M_r.back() < 1.0) {
        PARTHENON_THROW("The mass of the star (" + std::to_string(M_r.back()*mass_unit/M_SUN) + " M_sun) is less than the requested black hole mass (" + std::to_string(M_BH) + " M_sun).");
    }

    const Real r_max = r.back();
    const Real r_min = r.front();

    
    DeviceInterpolator1D rho_interp(r, rho_r);
    DeviceInterpolator1D Omega_interp(r, Omega_r);
    DeviceInterpolator1D u_interp(r, u_r);
    DeviceInterpolator1D M_interp(r, M_r);

    // Find the radius where the mass of the star is equal to the requested black hole mass, or equivalently, the core radius where the mass enclosed
    Real r_core = bisect([&](Real r) { return M_interp(r) - 1.0; },r_min, r_max, 0.0);


    if (MPIRank0() && pmb->gid == 0){
        std::cout << "rmin = " << r_min << " (in code units)" << std::endl;
        std::cout << "rmax = " << r_max << " (in code units)" << std::endl;
        std::cout << "length unit = " << length_unit << " cm" << std::endl;
        std::cout << "Core radius = " << r_core << " (in code units)" << std::endl;
        std::cout << "Core radius = " << r_core*length_unit << " cm" << std::endl;
        std::cout << "Mass of the star = " << M_interp(r_max) * mass_unit / M_SUN << " M_sun" << std::endl;
    }


    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("collapsar_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (int k, int j, int i) {
            Real X[GR_DIM];
            GR.coord_embed(k, j, i, Loci::center, X);
            const GReal r = X[1];
            if (r <= r_max && r >= r_core) {

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

    Floors::ApplyInitialFloors(pin, rc.get(), IndexDomain::interior);
    return TaskStatus::complete;
}