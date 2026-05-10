#include <iostream>
#include <cmath>
#include "solver.hpp"

// Device functions for dynamics and RK4 step
__device__ void system_dynamics(const double* state, double* deriv, double alpha) {
    double theta = state[0], phi = state[1], l1 = state[2], l2 = state[3];
    deriv[0] = phi;
    deriv[1] = sin(theta) - alpha * phi - l2 * cos(theta) * cos(theta);
    deriv[2] = -sin(theta) - l2 * cos(theta) - l2 * l2 * sin(theta) * cos(theta);
    deriv[3] = -phi - l1 + alpha * l2;
}

__device__ void rk4_step(double* state, double dt, double alpha) {
    double k1[4], k2[4], k3[4], k4[4], temp[4];
    system_dynamics(state, k1, alpha);
    for(int i=0; i<4; i++) temp[i] = state[i] + 0.5 * dt * k1[i];
    system_dynamics(temp, k2, alpha);
    for(int i=0; i<4; i++) temp[i] = state[i] + 0.5 * dt * k2[i];
    system_dynamics(temp, k3, alpha);
    for(int i=0; i<4; i++) temp[i] = state[i] + dt * k3[i];
    system_dynamics(temp, k4, alpha);
    for(int i=0; i<4; i++) state[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

// GPU Kernel to evaluate multiple guesses across multiple sheets
__global__ void multi_sheet_shooting_kernel(double theta_base, double phi0, double alpha,
                                            double P00, double P01, double P10, double P11,
                                            double* best_l1, double* best_l2, double* best_cost) {
    int k = blockIdx.x - 2;
    double theta0 = theta_base + 2.0 * M_PI * k;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // LQR guess for this sheet
    double l1_guess = P00 * theta0 + P01 * phi0;
    double l2_guess = P10 * theta0 + P11 * phi0;

    // Add small perturbations
    double perturb_scale = 0.05;
    double l1_test = l1_guess + perturb_scale * ((double)(tid % 16 - 8) / 8.0);
    double l2_test = l2_guess + perturb_scale * ((double)(tid / 16 - 8) / 8.0);

    double state[4] = {theta0, phi0, l1_test, l2_test};
    double dt = 0.05;
    double cost = 0.0;

    for(int step = 0; step < 400; step++) {
        double u = -state[3] * cos(state[0]);
        cost += ((1.0 - cos(state[0])) + 0.5 * state[1]*state[1] + 0.5 * u*u) * dt;
        rk4_step(state, dt, alpha);

        if (isnan(state[0]) || isnan(state[1]) || isnan(state[2]) || isnan(state[3])) {
            cost = 1e15;
            break;
        }
    }

    if (!isnan(cost) && cost < 1e14) {
         cost += 100.0 * (state[0]*state[0] + state[1]*state[1]);
    }

    int out_idx = blockIdx.x * num_threads + threadIdx.x;
    best_l1[out_idx] = l1_test;
    best_l2[out_idx] = l2_test;
    best_cost[out_idx] = cost;
}

// Analytical LQR Solver (No Eigen needed)
void solveARE(double alpha, double& P00, double& P01, double& P10, double& P11) {
    // Solving A^T P + P A - P B B^T P + Q = 0 analytically
    // A = [0 1; 1 -alpha], B = [0; 1], Q = [1 0; 0 1]
    double p12 = 1.0 + sqrt(2.0);
    double p22 = -alpha + sqrt(alpha * alpha + 2.0 * p12 + 1.0);
    double p11 = alpha * p12 + p22 * (p12 - 1.0);
    
    P00 = p11;
    P01 = p12;
    P10 = p12;
    P11 = p22;
}

Result solve(double theta, double phi, double alpha) {
    double P00, P01, P10, P11;
    solveARE(alpha, P00, P01, P10, P11);

    int num_sheets = 5;
    int threads_per_sheet = 256;
    int total_evals = num_sheets * threads_per_sheet;

    double *d_l1, *d_l2, *d_cost;
    cudaMalloc(&d_l1, total_evals * sizeof(double));
    cudaMalloc(&d_l2, total_evals * sizeof(double));
    cudaMalloc(&d_cost, total_evals * sizeof(double));

    multi_sheet_shooting_kernel<<<num_sheets, threads_per_sheet>>>(
        theta, phi, alpha, P00, P01, P10, P11, d_l1, d_l2, d_cost
    );
    cudaDeviceSynchronize();

    double *h_l1 = new double[total_evals];
    double *h_l2 = new double[total_evals];
    double *h_cost = new double[total_evals];

    cudaMemcpy(h_l1, d_l1, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l2, d_l2, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, total_evals * sizeof(double), cudaMemcpyDeviceToHost);

    int best_idx = -1;
    double min_cost = 1e20;
    for(int i = 0; i < total_evals; i++) {
        if(!isnan(h_cost[i]) && h_cost[i] < min_cost) {
            min_cost = h_cost[i];
            best_idx = i;
        }
    }

    Result r;
    if (best_idx >= 0) {
        r.l1 = h_l1[best_idx];
        r.l2 = h_l2[best_idx];
        r.cost = min_cost;
    } else {
        r.l1 = 0;
        r.l2 = 0;
        r.cost = 1e20;
    }

    cudaFree(d_l1); cudaFree(d_l2); cudaFree(d_cost);
    delete[] h_l1; delete[] h_l2; delete[] h_cost;

    return r;
}
