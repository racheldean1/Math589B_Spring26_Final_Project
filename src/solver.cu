#include <iostream>
#include <cmath>
#include <cfloat>
#include "solver.hpp"

#define THREADS 256
#define BLOCKS 4096

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

__global__ void global_grid_search(double theta_input, double phi_input, double alpha,
                                   double l1_min, double l1_max, double l2_min, double l2_max,
                                   double* best_l1, double* best_l2, double* min_cost) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    int grid_side = sqrt((double)total_threads);
    int row = tid / grid_side;
    int col = tid % grid_side;
    
    double l1 = l1_min + (l1_max - l1_min) * ((double)row / (grid_side - 1));
    double l2 = l2_min + (l2_max - l2_min) * ((double)col / (grid_side - 1));
    
    double state[4] = {theta_input, phi_input, l1, l2};
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
    
    if (cost < 1e14) {
        cost += 1000.0 * (state[0]*state[0] + state[1]*state[1]);
    }
    
    best_l1[tid] = l1;
    best_l2[tid] = l2;
    min_cost[tid] = cost;
}

Result solve(double theta, double phi, double alpha) {
    int total_evals = BLOCKS * THREADS;
    double *d_l1, *d_l2, *d_cost;
    cudaMalloc(&d_l1, total_evals * sizeof(double));
    cudaMalloc(&d_l2, total_evals * sizeof(double));
    cudaMalloc(&d_cost, total_evals * sizeof(double));
    
    double search_radius = 50.0;
    
    // Pass 1: Global coarse search
    global_grid_search<<<BLOCKS, THREADS>>>(theta, phi, alpha, -search_radius, search_radius, -search_radius, search_radius, d_l1, d_l2, d_cost);
    cudaDeviceSynchronize();
    
    double *h_l1 = new double[total_evals];
    double *h_l2 = new double[total_evals];
    double *h_cost = new double[total_evals];
    
    cudaMemcpy(h_l1, d_l1, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l2, d_l2, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    
    int best_idx = 0;
    for(int i = 1; i < total_evals; i++) {
        if(h_cost[i] < h_cost[best_idx]) best_idx = i;
    }
    
    double best_l1_val = h_l1[best_idx];
    double best_l2_val = h_l2[best_idx];
    
    // Pass 2: Fine search around the best candidate
    double fine_radius = 2.0;
    global_grid_search<<<BLOCKS, THREADS>>>(theta, phi, alpha, best_l1_val - fine_radius, best_l1_val + fine_radius, best_l2_val - fine_radius, best_l2_val + fine_radius, d_l1, d_l2, d_cost);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_l1, d_l1, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l2, d_l2, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
    
    best_idx = 0;
    for(int i = 1; i < total_evals; i++) {
        if(h_cost[i] < h_cost[best_idx]) best_idx = i;
    }
    
    Result r;
    r.l1 = h_l1[best_idx];
    r.l2 = h_l2[best_idx];
    r.cost = h_cost[best_idx];
    
    cudaFree(d_l1); cudaFree(d_l2); cudaFree(d_cost);
    delete[] h_l1; delete[] h_l2; delete[] h_cost;
    
    return r;
}
