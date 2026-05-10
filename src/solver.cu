#include <iostream>
#include <cmath>
#include <cfloat>
#include "solver.hpp"

#define NUM_SHEETS 5
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_SHEET 1024 // 262,144 guesses per sheet

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

__global__ void massive_shooting_kernel(double theta_input, double phi_input, double alpha,
                                        double P00, double P01, double P10, double P11,
                                        double* best_l1, double* best_l2, double* min_cost) {
    int sheet_idx = blockIdx.y;
    int k = sheet_idx - (NUM_SHEETS / 2);
    
    // Normalize theta to [-pi, pi)
    double theta_norm = fmod(theta_input + M_PI, 2.0 * M_PI);
    if (theta_norm < 0) theta_norm += 2.0 * M_PI;
    theta_norm -= M_PI;
    
    double target_theta = theta_norm + 2.0 * M_PI * k;
    
    // LQR guess for this sheet
    double l1_guess = P00 * target_theta + P01 * phi_input;
    double l2_guess = P10 * target_theta + P11 * phi_input;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Create a 2D grid of perturbations around the LQR guess
    int grid_side = sqrt((double)total_threads);
    int row = tid / grid_side;
    int col = tid % grid_side;
    
    // Spread perturbations: +/- 5.0 around the guess (can be tuned)
    double max_perturb = 5.0;
    double dl1 = -max_perturb + 2.0 * max_perturb * ((double)row / (grid_side - 1));
    double dl2 = -max_perturb + 2.0 * max_perturb * ((double)col / (grid_side - 1));
    
    double state[4] = {theta_input, phi_input, l1_guess + dl1, l2_guess + dl2};
    double dt = 0.05;
    double cost = 0.0;
    
    for(int step = 0; step < 500; step++) {
        double u = -state[3] * cos(state[0]);
        cost += ((1.0 - cos(state[0])) + 0.5 * state[1]*state[1] + 0.5 * u*u) * dt;
        rk4_step(state, dt, alpha);
        
        if (isnan(state[0]) || isnan(state[1]) || isnan(state[2]) || isnan(state[3])) {
            cost = 1e15;
            break;
        }
    }
    
    // Penalize failure to reach origin
    if (cost < 1e14) {
        cost += 1000.0 * (state[0]*state[0] + state[1]*state[1]);
    }
    
    int out_idx = sheet_idx * total_threads + tid;
    best_l1[out_idx] = l1_guess + dl1;
    best_l2[out_idx] = l2_guess + dl2;
    min_cost[out_idx] = cost;
}

void solveARE(double alpha, double& P00, double& P01, double& P10, double& P11) {
    double p12 = 1.0 + sqrt(2.0);
    double p22 = -alpha + sqrt(alpha * alpha + 2.0 * p12 + 1.0);
    double p11 = alpha * p12 + p22 * (p12 - 1.0);
    P00 = p11; P01 = p12; P10 = p12; P11 = p22;
}

Result solve(double theta, double phi, double alpha) {
    double P00, P01, P10, P11;
    solveARE(alpha, P00, P01, P10, P11);
    
    int threads = THREADS_PER_BLOCK;
    int blocks_x = BLOCKS_PER_SHEET;
    int sheets = NUM_SHEETS;
    int total_evals = sheets * blocks_x * threads;
    
    double *d_l1, *d_l2, *d_cost;
    cudaMalloc(&d_l1, total_evals * sizeof(double));
    cudaMalloc(&d_l2, total_evals * sizeof(double));
    cudaMalloc(&d_cost, total_evals * sizeof(double));
    
    dim3 grid(blocks_x, sheets);
    massive_shooting_kernel<<<grid, threads>>>(theta, phi, alpha, P00, P01, P10, P11, d_l1, d_l2, d_cost);
    cudaDeviceSynchronize();
    
    double *h_l1 = new double[total_evals];
    double *h_l2 = new double[total_evals];
    double *h_cost = new double[total_evals];
    
    cudaMemcpy(h_l1, d_l1, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l2, d_l2, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    
    int best_idx = -1;
    double min_c = 1e20;
    for(int i = 0; i < total_evals; i++) {
        if(!isnan(h_cost[i]) && h_cost[i] < min_c) {
            min_c = h_cost[i];
            best_idx = i;
        }
    }
    
    Result r;
    if(best_idx >= 0) {
        r.l1 = h_l1[best_idx];
        r.l2 = h_l2[best_idx];
        r.cost = min_c;
    } else {
        r.l1 = 0; r.l2 = 0; r.cost = 1e20;
    }
    
    cudaFree(d_l1); cudaFree(d_l2); cudaFree(d_cost);
    delete[] h_l1; delete[] h_l2; delete[] h_cost;
    return r;
}
