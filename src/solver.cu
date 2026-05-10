#include <iostream>
#include <cmath>
#include <cfloat>
#include "solver.hpp"

#define NUM_SHEETS 5
#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_SHEET 256

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

__device__ void evaluate_F(double l1, double l2, double theta0, double phi0, double alpha, double& f1, double& f2, double& out_cost) {
    double state[4] = {theta0, phi0, l1, l2};
    double dt = 0.05;
    out_cost = 0.0;
    for(int step=0; step<400; step++) {
        double u = -state[3]*cos(state[0]);
        out_cost += ((1.0-cos(state[0])) + 0.5*state[1]*state[1] + 0.5*u*u)*dt;
        rk4_step(state, dt, alpha);
    }
    f1 = state[0];
    f2 = state[1];
}

__global__ void refined_shooting_kernel(double theta_input, double phi_input, double alpha,
                                        double P00, double P01, double P10, double P11,
                                        double* best_l1, double* best_l2, double* min_cost) {
    int sheet_idx = blockIdx.y;
    int k = sheet_idx - (NUM_SHEETS / 2);

    double theta_norm = fmod(theta_input + M_PI, 2.0 * M_PI);
    if (theta_norm < 0) theta_norm += 2.0 * M_PI;
    theta_norm -= M_PI;

    double target_theta = theta_norm + 2.0 * M_PI * k;

    double l1_guess = P00 * target_theta + P01 * phi_input;
    double l2_guess = P10 * target_theta + P11 * phi_input;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    int grid_side = sqrt((double)total_threads);
    int row = tid / grid_side;
    int col = tid % grid_side;

    double max_perturb = 8.0;
    double dl1 = -max_perturb + 2.0 * max_perturb * ((double)row / (grid_side - 1));
    double dl2 = -max_perturb + 2.0 * max_perturb * ((double)col / (grid_side - 1));

    double opt_l1 = l1_guess + dl1;
    double opt_l2 = l2_guess + dl2;
    
    double f1, f2, cost;
    evaluate_F(opt_l1, opt_l2, theta_input, phi_input, alpha, f1, f2, cost);
    double current_err = f1*f1 + f2*f2;

    double lr = 1e-3;
    for(int iter=0; iter<50; iter++) {
        if(current_err < 1e-8 || isnan(current_err)) break;

        double eps = 1e-5;
        double f1_l1, f2_l1, c_dummy;
        evaluate_F(opt_l1 + eps, opt_l2, theta_input, phi_input, alpha, f1_l1, f2_l1, c_dummy);
        double err_l1 = f1_l1*f1_l1 + f2_l1*f2_l1;
        double J_l1 = (err_l1 - current_err) / eps;

        double f1_l2, f2_l2;
        evaluate_F(opt_l1, opt_l2 + eps, theta_input, phi_input, alpha, f1_l2, f2_l2, c_dummy);
        double err_l2 = f1_l2*f1_l2 + f2_l2*f2_l2;
        double J_l2 = (err_l2 - current_err) / eps;

        double next_l1 = opt_l1 - lr * J_l1;
        double next_l2 = opt_l2 - lr * J_l2;

        double next_f1, next_f2, next_cost;
        evaluate_F(next_l1, next_l2, theta_input, phi_input, alpha, next_f1, next_f2, next_cost);
        double next_err = next_f1*next_f1 + next_f2*next_f2;

        if (isnan(next_err) || next_err >= current_err) {
            lr *= 0.5;
        } else {
            opt_l1 = next_l1;
            opt_l2 = next_l2;
            current_err = next_err;
            cost = next_cost;
            lr *= 1.2;
        }
    }

    if (current_err > 1e-3) {
        cost += 1000.0 * current_err;
    }

    int out_idx = sheet_idx * total_threads + tid;
    best_l1[out_idx] = opt_l1;
    best_l2[out_idx] = opt_l2;
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
    refined_shooting_kernel<<<grid, threads>>>(theta, phi, alpha, P00, P01, P10, P11, d_l1, d_l2, d_cost);
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
