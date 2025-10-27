
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include "pnetcdf.h" // NetCDF library for I/O
#include <chrono>

// Define physical constants for weather simulation
// constexpr: a compile-time constant, cannot be changed at runtime, but can be modified by the compiler
// so it is more efficient than #define, and it is more type-safe than #define.
constexpr double pi        = 3.14159265358979323846264338327;   //Pi
constexpr double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
constexpr double cp        = 1004.;                             //Specific heat of dry air at constant pressure
constexpr double cv        = 717.;                              //Specific heat of dry air at constant volume
constexpr double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr double C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr double gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)


//Define domain and stability-related constants
constexpr double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
constexpr double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
constexpr double hv_beta   = 0.05;    //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability), 一个时间步长内，信息传播的距离不能超过网格间距的1.5倍
constexpr double max_speed = 450;     //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction， 避免边界特殊处理
constexpr int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4;           //Number of fluid state variables， 密度、动量、势能、压力
constexpr int ID_DENS  = 0;           //index for density ("rho")
constexpr int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")

constexpr int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
constexpr int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction

// this is the data specification for the simulation, different data initialization methods
// these method are used to initialize the fluid state, different initial conditions
constexpr int DATA_SPEC_COLLISION       = 1; // 碰撞模型
constexpr int DATA_SPEC_THERMAL         = 2; // 热力学模型
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3; // 重力波模型
constexpr int DATA_SPEC_DENSITY_CURRENT = 5; // 密度当前模型
constexpr int DATA_SPEC_INJECTION       = 6; // 注入模型

// variables for quadrature integration
constexpr int nqpoints = 3; // Number of quadrature points for integration
// Quadrature points and weights for integration, fixed for now
constexpr double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
constexpr double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// BEGIN USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////
//The x-direction length is twice as long as the z-direction length
//So, you'll want to have nx_glob be twice as large as nz_glob
int    constexpr nx_glob       = _NX;            //Number of total cells in the x-direction
int    constexpr nz_glob       = _NZ;            //Number of total cells in the z-direction
double constexpr sim_time      = _SIM_TIME;      //How many seconds to run the simulation
double constexpr output_freq   = _OUT_FREQ;      //How frequently to output data to file (in seconds)
int    constexpr data_spec_int = _DATA_SPEC;     //How to initialize the data
double constexpr dx            = xlen / nx_glob; // grid spacing in the x-direction
double constexpr dz            = zlen / nz_glob; // grid spacing in the x-direction
///////////////////////////////////////////////////////////////////////////////////////
// END USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double dt;                    //Model time step (seconds)
int    nx, nz;                //Number of local grid cells in the x- and z- dimensions for this MPI task
int    i_beg, k_beg;          //beginning index in the x- and z-directions for this MPI task
int    nranks, myrank;        //Number of MPI ranks and my rank id
int    left_rank, right_rank; //MPI Rank IDs that exist to my left and right in the global domain
int    mainproc;              //Am I the main process (rank == 0)?
double *hy_dens_cell;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
double *hy_dens_theta_cell;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
double *hy_dens_int;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
double *hy_dens_theta_int;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
double *hy_pressure_int;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;                 //Elapsed model time
double output_counter;        //Helps determine when it's time to do output
// The main variables for the model
double *state;                //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS), three dimensions: (x, z, variables)
double *state_tmp;            //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS), three dimensions: (x, z, variables)
double *flux;                 //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS), three dimensions: (x, z, variables)
double *tend;                 //Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS), three dimensions: (x, z, variables)
// the following variables are used to track the simulation progress
int    num_out = 0;           //The number of outputs performed so far
int    direction_switch = 1;
double mass0, te0;            //Initial domain totals for mass and total energy  
double mass , te ;            //Domain totals for mass and total energy  

//How is this not in the standard?!
// This is a helper function to find the minimum of two values
double dmin( double a , double b ) { if (a<b) {return a;} else {return b;} };


//Declaring the functions defined after "main"
void   init                 ( int *argc , char ***argv );
void   finalize             ( );
void   injection            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   density_current      ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   gravity_waves        ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   thermal              ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   collision            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   hydro_const_theta    ( double z                   , double &r , double &t );
void   hydro_const_bvfreq   ( double z , double bv_freq0 , double &r , double &t );
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad );
void   output               ( double *state , double etime );
void   ncwrap               ( int ierr , int line );
void   perform_timestep     ( double *state , double *state_tmp , double *flux , double *tend , double dt );
void   semi_discrete_step   ( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend );
void   compute_tendencies_x ( double *state , double *flux , double *tend , double dt);
void   compute_tendencies_z ( double *state , double *flux , double *tend , double dt);
void   set_halo_values_x    ( double *state );
void   set_halo_values_z    ( double *state );
void   reductions           ( double &mass , double &te );


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  // MPI initialization
  init( &argc , &argv );

  //Initial reductions for mass, kinetic energy, and total energy
  // 在 mpi 环境下，需要进行全局归约，计算初始时刻的 mass 和 te
  reductions(mass0,te0);

  //Output the initial state
  // 如果输出频率大于0，则输出初始状态
  if (output_freq >= 0) output(state,etime);

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  auto t1 = std::chrono::steady_clock::now(); // Start timing the simulation
  while (etime < sim_time) { // Main time step loop
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    // 截断时间步长，确保不超出模拟时间
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    // 主要计算步骤，包括计算通量、计算倾向、更新状态
    perform_timestep(state,state_tmp,flux,tend,dt); // Perform a single time step
    //Inform the user
    // 输出模拟进度
#ifndef NO_INFORM
    if (mainproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); } // Inform the user of the simulation progress
#endif
    //Update the elapsed time and output counter
    etime = etime + dt; // Update the elapsed time
    output_counter = output_counter + dt; // Update the output counter
    //If it's time for output, reset the counter, and do output
    if (output_freq >= 0 && output_counter >= output_freq) {
      output_counter = output_counter - output_freq; // Reset the output counter
      output(state,etime); // Output the state
    }
  }
  auto t2 = std::chrono::steady_clock::now(); // Stop timing the simulation
  if (mainproc) {
    std::cout << "CPU Time: " << std::chrono::duration<double>(t2-t1).count() << " sec\n"; // Print the simulation time
  }

  //Final reductions for mass, kinetic energy, and total energy
  reductions(mass,te); // Final reductions for mass, kinetic energy, and total energy

  if (mainproc) {
    printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
    printf( "d_te:   %le\n" , (te   - te0  )/te0   );
  }

  finalize(); // Deallocate and finialize MPI
}


//Performs a single dimensionally split time step using a simple low-storage three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep( double *state , double *state_tmp , double *flux , double *tend , double dt ) {
  if (direction_switch) {
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
  } else {
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}

// 用于计算一个时间步长，包括 x 方向和 z 方向
//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend ) {
  int i, k, ll, inds, indt, indw;
  double x, z, wpert, dist, x0, z0, xrad, zrad, amp;
  if        (dir == DIR_X) { // x-direction
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(state_forcing); // 设置边界值
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(state_forcing,flux,tend,dt);
  } else if (dir == DIR_Z) { // z-direction
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(state_forcing);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(state_forcing,flux,tend,dt);
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
          x = (i_beg + i+0.5)*dx;
          z = (k_beg + k+0.5)*dz;
          // Using sample_ellipse_cosine requires "acc routine" in OpenACC and "declare target" in OpenMP offload
          // Neither of these are particularly well supported. So I'm manually inlining here
          // wpert = sample_ellipse_cosine( x,z , 0.01 , xlen/8,1000., 500.,500. );
          {
            x0   = xlen/8;
            z0   = 1000;
            xrad = 500;
            zrad = 500;
            amp  = 0.01;
            //Compute distance from bubble center
            dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
            //If the distance from bubble center is less than the radius, create a cos**2 profile
            if (dist <= pi / 2.) {
              wpert = amp * pow(cos(dist),2.);
            } else {
              wpert = 0.;
            }
          }
          indw = ID_WMOM*nz*nx + k*nx + i;
          tend[indw] += wpert*hy_dens_cell[hs+k];
        }
        inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs; // calculate the index of the fluid state
        indt = ll*nz*nx + k*nx + i; // calculate the index of the tendency
        state_out[inds] = state_init[inds] + dt * tend[indt]; // apply the tendencies to the fluid state
      }
    }
  }
}

// 用于计算 x 方向的倾向，由于 halos 在 set_halo_values_x 中已经设置，所以不需要 MPI
//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x( double *state , double *flux , double *tend , double dt) {
  int    i,k,ll,s,inds,indf1,indf2,indt;
  double r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coefficient
  hv_coef = -hv_beta * dx / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz; k++) {
    for (i=0; i<nx+1; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s < sten_size; s++) {
          inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
          stencil[s] = state[inds];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_cell[k+hs];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
      p = C0*pow((r*t),gamm);

      //Compute the flux vector
      flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        indt  = ll* nz   * nx    + k* nx    + i  ;
        indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
        indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
        tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
      }
    }
  }
}

// 用于计算 z 方向的倾向，由于没有 MPI 分解，所以不需要 MPI
//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z( double *state , double *flux , double *tend , double dt) {
  int    i,k,ll,s, inds, indf1, indf2, indt;
  double r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coefficient
  hv_coef = -hv_beta * dz / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nz+1; k++) {
    for (i=0; i<nx; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s<sten_size; s++) {
          inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
          stencil[s] = state[inds];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_int[k];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
      p = C0*pow((r*t),gamm) - hy_pressure_int[k];
      //Enforce vertical boundary condition and exact mass conservation
      if (k == 0 || k == nz) {
        w                = 0;
        d3_vals[ID_DENS] = 0;
      }

      //Compute the flux vector with hyperviscosity
      flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      for (i=0; i<nx; i++) {
        indt  = ll* nz   * nx    + k* nx    + i  ;
        indf1 = ll*(nz+1)*(nx+1) + (k  )*(nx+1) + i;
        indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
        tend[indt] = -( flux[indf2] - flux[indf1] ) / dz;
        if (ll == ID_WMOM) {
          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
          tend[indt] = tend[indt] - state[inds]*grav;
        }
      }
    }
  }
}


// 用于设置 MPI 任务的边界值，在 x 方向上，需要 MPI 通信
//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x( double *state ) {
  int k, ll, ind_r, ind_u, ind_t, i;
  double z;
  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nx-hs+1:nx,1:nz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nx+1:nx+hs,1:nz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nz; k++) {
      state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + 0      ] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs-2];
      state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + 1      ] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs-1];
      state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs  ] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs     ];
      state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs+1] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs+1   ];
    }
  }
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      for (k=0; k<nz; k++) {
        for (i=0; i<hs; i++) {
          z = (k_beg + k+0.5)*dz;
          if (fabs(z-3*zlen/4) <= zlen/16) {
            ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
            state[ind_u] = (state[ind_r]+hy_dens_cell[k+hs]) * 50.;
            state[ind_t] = (state[ind_r]+hy_dens_cell[k+hs]) * 298. - hy_dens_theta_cell[k+hs];
          }
        }
      }
    }
  }
}

// 用于设置 MPI 任务的边界值，在 z 方向上，由于没有 MPI 分解，所以不需要 MPI
//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z( double *state ) {
  int          i, ll;
  const double mnt_width = xlen/8;
  double       x, xloc, mnt_deriv;
  /////////////////////////////////////////////////
  // TODO: THREAD ME 线程并行
  /////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) { // 遍历所有变量
    for (i=0; i<nx+2*hs; i++) { // 遍历所有单元
      if (ll == ID_WMOM) { // z 方向垂直风速
        state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = 0.;
        state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = 0.;
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = 0.;
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = 0.;
      } else if (ll == ID_UMOM) { // x 方向水平风速
        state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i] / hy_dens_cell[hs     ] * hy_dens_cell[0      ];
        state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i] / hy_dens_cell[hs     ] * hy_dens_cell[1      ];
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i] / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs  ];
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i] / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs+1];
      } else { // 其他变量
        state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
        state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
        state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
      }
    }
  }
}


/**
 * 1. MPI Initialization
 * 2. allocate memory for the model data
 * 3. time step initialization
 * 4. set the initial conditions
 * 5. 静水平衡背景场计算
 */
void init( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, ierr, inds;
  double x, z, r, u, w, t, hr, ht;

  // MPI initialization
  ierr = MPI_Init(argc,argv);

  /////////////////////////////////////////////////////////////
  // BEGIN MPI DUMMY SECTION: for the serial version on x-direction
  // TODO: (1) GET NUMBER OF MPI RANKS
  //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  /////////////////////////////////////////////////////////////
  nranks = 1; // number of MPI ranks
  myrank = 0; // my MPI rank id
  i_beg = 0; // my beginning "I" index
  nx = nx_glob; // my number of X-direction cells
  left_rank = 0; // my left neighboring rank id
  right_rank = 0; // my right neighboring rank id
  //////////////////////////////////////////////
  // END MPI DUMMY SECTION
  //////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  // z 方向不进行 MPI 分解，因为大气模拟通常在水平方向更大，垂直方向的物理过程需要整列数据
  k_beg = 0;
  nz = nz_glob;
  mainproc = (myrank == 0);

  // 流体状态数组
  state              = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp          = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  // 通量和趋势数组
  flux               = (double *) malloc( (nx+1)*(nz+1)*NUM_VARS*sizeof(double) );
  tend               = (double *) malloc( nx*nz*NUM_VARS*sizeof(double) );
  // 静水平衡背景场计算所需的变量
  hy_dens_cell       = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_theta_cell = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_int        = (double *) malloc( (nz+1)*sizeof(double) );
  hy_dens_theta_int  = (double *) malloc( (nz+1)*sizeof(double) );
  hy_pressure_int    = (double *) malloc( (nz+1)*sizeof(double) );

  //Define the maximum stable time step based on an assumed maximum wind speed
  // 最大稳定时间步长计算，根据最大风速和 CFL 条件 -> 保证数值稳定性
  // dx, dy 是 grid space，max_speed 是最大风速，cfl 是 CFL 数
  dt = dmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  //If I'm the main process in MPI, display some grid information
  if (mainproc) {
    printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }

  // 显式同步，确保所有 MPI 进程都到达这里
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  // 对每个单元初始化流体状态
  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      //初始化该 cell 的流体状态为零
      for (ll=0; ll<NUM_VARS; ll++) {
        // 内存排布是：(nz+2*hs)*(nx+2*hs)*NUM_VARS，每个变量一个数组
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      // 3 x 3 积分点
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          // 计算积分点在全局域中的位置，基于单元和积分点索引
          // 单元 i 的中心：x = i*dx + 0.5*dx，同理 z = k*dz + 0.5*dz
          // 积分点：x₁ = 中心 + (x' - 0.5)*dx
          center_x = (i_beg + i-hs+0.5)*dx;
          center_z = (k_beg + k-hs+0.5)*dz;
          x = center_x + (qpoints[ii]-0.5)*dx;
          z = center_z + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          // here wiil initialize the fluid state
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          // 累加加权, the value of r, hr, ht, u, w, t are initialized in the function collision, thermal, gravity_waves, density_current, injection
          // ID_ is the index of the variable in the state array
          // 这里计算的是积分点的值，需要累加到单元平均值
          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + r                         * qweights[ii]*qweights[kk];
          inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      // 当前单元的物理状态，计算全局 index，复制流体状态到临时数组
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state_tmp[inds] = state[inds];
      }
      // 移动到下一个单元
    }
  }

  // 计算单元平均背景场，用于计算通量
  //Compute the hydrostatic background state over vertical cell averages
  for (k=0; k<nz+2*hs; k++) {
    hy_dens_cell      [k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      [k] = hy_dens_cell      [k] + hr    * qweights[kk]; // 单元平均
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  // 计算单元界面背景场，用于边界重构
  //Compute the hydrostatic background state at vertical cell interfaces
  for (k=0; k<nz+1; k++) {
    z = (k_beg + k)*dz;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr*ht;
    hy_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }
}

// 注入，从左边界注入冷气，这是最简单的初始条件，没有密度流，没有重力波，没有热气泡，没有碰撞热气泡
//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;// 密度为零
  t = 0.;// 温度为零
  u = 0.;// 风速为零
  w = 0.;// 风速为零
}

// 密度流
//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void density_current( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;// 密度为零
  t = 0.;// 温度为零
  u = 0.; // 风速为零
  w = 0.;// 风速为零
  t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.); // 在密度当前中心附近有一个冷气泡（-20K）
}

// 重力波
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void gravity_waves( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_bvfreq(z,0.02,hr,ht); // 计算静水平衡背景场， 0.02 是 Brunt-Vaisala 频率
  r = 0.;// 密度为零
  t = 0.;// 温度为零
  u = 15.;// 风速为15 m/s
  w = 0.;// 风速为零
}

//Rising thermal: 热气泡
// 由于浮力，气泡会上升，在上升过程中，温度会降低，密度会增大
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void thermal( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);// 计算静水平衡背景场
  r = 0.;// 密度为零
  t = 0.;// 温度为零
  u = 0.;// 风速为零
  w = 0.;// 风速为零
  t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);// 在热气泡中心附近，温度增加3K
  // sample_ellipse_cosine 是一个椭圆函数，用于在热气泡中心附近，温度增加3K
  // xlen/2 是热气泡中心的x坐标
  // 2000. 是热气泡的半径
  // 2000. 是热气泡的高度
  // 2000. 是热气泡的深度
  // 3. 是热气泡的温度增加量
}

// 碰撞热气泡
//Colliding thermals
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void collision( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht); // 计算静水平衡背景场
  r = 0.;// 密度为零
  t = 0.;// 温度为零
  u = 0.;// 风速为零
  w = 0.;// 风速为零
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);// 在碰撞热气泡中心附近，温度增加20K
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);// 在碰撞热气泡中心附近，温度降低20K
  // 有两个热气泡，一个在2000m高度，一个在8000m高度，温度分别为20K和-20K
  // 冷气泡会下沉，热气泡会上升，形成对流，发生碰撞，形成新的热气泡
}
/**
 * 系统完整状态：背景场 + 扰动场
 * 背景场：静水平衡背景场，不随时间变化的物理量，包括密度、温度、风速等
 * 扰动场：随时间变化的物理量，包括密度、温度、风速等
 * 背景场和扰动场共同构成了系统完整状态
 */

// 计算静水平衡背景场，使用常数势温度（热力中性大气）
//Establish hydrostatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}

// 计算静水平衡背景场，使用常数 Brunt-Vaisala 频率，用于重力波模拟
//Establish hydrostatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                     //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}

// 计算椭圆函数，用于初始化热气泡、密度当前、碰撞热气泡等
//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}

// 用于输出模拟结果到 NetCDF 文件，可视化结果
//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses parallel-netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
void output( double *state , double etime ) {
  int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
  int i, k, ind_r, ind_u, ind_w, ind_t;
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];
  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
  double *dens, *uwnd, *wwnd, *theta;
  double *etimearr;
  //Inform the user
  if (mainproc) { printf("*** OUTPUT ***\n"); }
  //Allocate some (big) temp arrays
  dens     = (double *) malloc(nx*nz*sizeof(double));
  uwnd     = (double *) malloc(nx*nz*sizeof(double));
  wwnd     = (double *) malloc(nx*nz*sizeof(double));
  theta    = (double *) malloc(nx*nz*sizeof(double));
  etimearr = (double *) malloc(1    *sizeof(double));

  //If the elapsed time is zero, create the file. Otherwise, open the file
  if (etime == 0) {
    //Create the file
    ncwrap( ncmpi_create( MPI_COMM_WORLD , "output.nc" , NC_CLOBBER , MPI_INFO_NULL , &ncid ) , __LINE__ );
    //Create the dimensions
    ncwrap( ncmpi_def_dim( ncid , "t" , (MPI_Offset) NC_UNLIMITED , &t_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "x" , (MPI_Offset) nx_glob      , &x_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "z" , (MPI_Offset) nz_glob      , &z_dimid ) , __LINE__ );
    //Create the variables
    dimids[0] = t_dimid;
    ncwrap( ncmpi_def_var( ncid , "t"     , NC_DOUBLE , 1 , dimids ,     &t_varid ) , __LINE__ );
    dimids[0] = t_dimid; dimids[1] = z_dimid; dimids[2] = x_dimid;
    ncwrap( ncmpi_def_var( ncid , "dens"  , NC_DOUBLE , 3 , dimids ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "uwnd"  , NC_DOUBLE , 3 , dimids ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "wwnd"  , NC_DOUBLE , 3 , dimids ,  &wwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "theta" , NC_DOUBLE , 3 , dimids , &theta_varid ) , __LINE__ );
    //End "define" mode
    ncwrap( ncmpi_enddef( ncid ) , __LINE__ );
  } else {
    //Open the file
    ncwrap( ncmpi_open( MPI_COMM_WORLD , "output.nc" , NC_WRITE , MPI_INFO_NULL , &ncid ) , __LINE__ );
    //Get the variable IDs
    ncwrap( ncmpi_inq_varid( ncid , "dens"  ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "uwnd"  ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "wwnd"  ,  &wwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "theta" , &theta_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "t"     ,     &t_varid ) , __LINE__ );
  }

  //Store perturbed values in the temp arrays for output
  for (k=0; k<nz; k++) {
    for (i=0; i<nx; i++) {
      ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      dens [k*nx+i] = state[ind_r];
      uwnd [k*nx+i] = state[ind_u] / ( hy_dens_cell[k+hs] + state[ind_r] );
      wwnd [k*nx+i] = state[ind_w] / ( hy_dens_cell[k+hs] + state[ind_r] );
      theta[k*nx+i] = ( state[ind_t] + hy_dens_theta_cell[k+hs] ) / ( hy_dens_cell[k+hs] + state[ind_r] ) - hy_dens_theta_cell[k+hs] / hy_dens_cell[k+hs];
    }
  }

  //Write the grid data to file with all the processes writing collectively
  st3[0] = num_out; st3[1] = k_beg; st3[2] = i_beg;
  ct3[0] = 1      ; ct3[1] = nz   ; ct3[2] = nx   ;
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta ) , __LINE__ );

  //Only the main process needs to write the elapsed time
  //Begin "independent" write mode
  ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
  //write elapsed time to file
  if (mainproc) {
    st1[0] = num_out;
    ct1[0] = 1;
    etimearr[0] = etime; ncwrap( ncmpi_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr ) , __LINE__ );
  }
  //End "independent" write mode
  ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );

  //Close the file
  ncwrap( ncmpi_close(ncid) , __LINE__ );

  //Increment the number of outputs
  num_out = num_out + 1;

  //Deallocate the temp arrays
  free( dens     );
  free( uwnd     );
  free( wwnd     );
  free( theta    );
  free( etimearr );
}


//Error reporting routine for the PNetCDF I/O
// 错误报告，用于 NetCDF 错误
void ncwrap( int ierr , int line ) {
  if (ierr != NC_NOERR) {
    printf("NetCDF Error at line: %d\n", line);
    printf("%s\n",ncmpi_strerror(ierr));
    exit(-1);
  }
}


// 释放内存，结束模拟
void finalize() {
  int ierr;
  free( state );
  free( state_tmp );
  free( flux );
  free( tend );
  free( hy_dens_cell );
  free( hy_dens_theta_cell );
  free( hy_dens_int );
  free( hy_dens_theta_int );
  free( hy_pressure_int );
  ierr = MPI_Finalize();
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
void reductions( double &mass , double &te ) {
  mass = 0;
  te   = 0;
  // 遍历所有单元，计算质量守恒和能量守恒
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      double r  =   state[ind_r] + hy_dens_cell[hs+k];             // Density
      double u  =   state[ind_u] / r;                              // U-wind
      double w  =   state[ind_w] / r;                              // W-wind
      double th = ( state[ind_t] + hy_dens_theta_cell[hs+k] ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      mass += r        *dx*dz; // Accumulate domain mass
      te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
  double glob[2], loc[2];
  loc[0] = mass;
  loc[1] = te;
  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  mass = glob[0];
  te   = glob[1];
}


