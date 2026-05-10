#include <iostream>
#include <cmath>
#include <cfloat>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "solver.hpp"

#define THREADS 256

// CUDA kernel for RK4 integration
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

__global__ void evaluate_sheets(double theta_base, double phi, double alpha,
                                double P00, double P01, double P10, double P11,
                                double* out_l1, double* out_l2, double* out_cost) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_offsets[5] = {-2, -1, 0, 1, 2};

    if (tid < 5) {
        double theta_adj = theta_base + k_offsets[tid] * 2.0 * M_PI;
        
        // Compute exact linear costates for this sheet using projection matrix P
        double l1 = P00 * theta_adj + P01 * phi;
        double l2 = P10 * theta_adj + P11 * phi;

        double state[4] = {theta_adj, phi, l1, l2};
        double dt = 0.05;
        double cost = 0.0;

        // Forward integration to evaluate the cost of this trajectory
        for(int step = 0; step < 400; step++) {
            double u = -state[3] * cos(state[0]);
            cost += ((1.0 - cos(state[0])) + 0.5 * state[1]*state[1] + 0.5 * u*u) * dt;
            rk4_step(state, dt, alpha);
        }

        out_l1[tid] = l1;
        out_l2[tid] = l2;
        out_cost[tid] = cost;
    }
}

Result solve(double theta, double phi, double alpha) {
    // 1. Solve the linearized Hamiltonian system using Eigen (Host side)
    Eigen::Matrix4d H;
    H << 0.0, 1.0, 0.0, 0.0,
         1.0, -alpha, 0.0, -1.0,
        -1.0, 0.0, 0.0, -1.0,
         0.0, -1.0, -1.0, alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(H);
    auto evals = es.eigenvalues();
    auto evecs = es.eigenvectors();

    Eigen::Matrix2cd Vs_x;
    Eigen::Matrix2cd Vs_l;
    int col = 0;
    
    // Find the stable manifold (eigenvectors for eigenvalues with Re < 0)
    for (int i = 0; i < 4; i++) {
        if (evals(i).real() < 0 && col < 2) {
            Vs_x(0, col) = evecs(0, i);
            Vs_x(1, col) = evecs(1, i);
            Vs_l(0, col) = evecs(2, i);
            Vs_l(1, col) = evecs(3, i);
            col++;
        }
    }

    // Projection matrix P = V_{s,l} * V_{s,x}^{-1}
    Eigen::Matrix2cd P_complex = Vs_l * Vs_x.inverse();
    Eigen::Matrix2d P = P_complex.real();

    // 2. Launch CUDA kernel to evaluate multiple sheets simultaneously
    double *d_l1, *d_l2, *d_cost;
    cudaMalloc(&d_l1, 5 * sizeof(double));
    cudaMalloc(&d_l2, 5 * sizeof(double));
    cudaMalloc(&d_cost, 5 * sizeof(double));

    evaluate_sheets<<<1, THREADS>>>(theta, phi, alpha, P(0,0), P(0,1), P(1,0), P(1,1), d_l1, d_l2, d_cost);
    cudaDeviceSynchronize();

    double h_l1[5], h_l2[5], h_cost[5];
    cudaMemcpy(h_l1, d_l1, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l2, d_l2, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, 5 * sizeof(double), cudaMemcpyDeviceToHost);

    int best_idx = 0;
    for(int i = 1; i < 5; i++) {
        if(h_cost[i] < h_cost[best_idx]) best_idx = i;
    }

    Result r;
    r.l1 = h_l1[best_idx];
    r.l2 = h_l2[best_idx];
    r.cost = h_cost[best_idx];

    cudaFree(d_l1); cudaFree(d_l2); cudaFree(d_cost);
    return r;
}
