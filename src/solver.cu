#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "solver.hpp"

#define THREADS 256
#define BLOCKS 256

__device__ __host__ void system_dynamics(const double* state, double* deriv, double alpha) {
    double theta = state[0], phi = state[1], l1 = state[2], l2 = state[3];
    deriv[0] = phi;
    deriv[1] = sin(theta) - alpha * phi - l2 * cos(theta) * cos(theta);
    deriv[2] = -sin(theta) - l2 * cos(theta) - l2 * l2 * sin(theta) * cos(theta);
    deriv[3] = -phi - l1 + alpha * l2;
}

// Backward RK4: dt is negative
__device__ __host__ void rk4_step_backward(double* state, double dt, double alpha, double& cost_accum) {
    double k1[4], k2[4], k3[4], k4[4], temp[4];

    auto calc_L = [](const double* s) {
        double u = -s[3] * cos(s[0]);
        return (1.0 - cos(s[0])) + 0.5 * s[1]*s[1] + 0.5 * u*u;
    };

    system_dynamics(state, k1, alpha);
    double c1 = calc_L(state);

    for(int i=0; i<4; i++) temp[i] = state[i] + 0.5 * dt * k1[i];
    system_dynamics(temp, k2, alpha);
    double c2 = calc_L(temp);

    for(int i=0; i<4; i++) temp[i] = state[i] + 0.5 * dt * k2[i];
    system_dynamics(temp, k3, alpha);
    double c3 = calc_L(temp);

    for(int i=0; i<4; i++) temp[i] = state[i] + dt * k3[i];
    system_dynamics(temp, k4, alpha);
    double c4 = calc_L(temp);

    for(int i=0; i<4; i++) state[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

    cost_accum += (fabs(dt) / 6.0) * (c1 + 2.0*c2 + 2.0*c3 + c4);
}

__global__ void coarse_search_kernel(double theta_eq, double theta_target, double phi_target, double alpha,
                                     double v1_0, double v1_1, double v1_2, double v1_3,
                                     double v2_0, double v2_1, double v2_2, double v2_3,
                                     double r, int grid_pts,
                                     double* out_a, double* out_b, double* out_res, double* out_cost, double* out_l1, double* out_l2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    double r_min = -r;
    double r_max = r;

    for(int i = tid; i < grid_pts * grid_pts; i += total_threads) {
        int row = i / grid_pts;
        int col = i % grid_pts;

        double a = r_min + (r_max - r_min) * ((double)row / (grid_pts - 1));
        double b = r_min + (r_max - r_min) * ((double)col / (grid_pts - 1));

        double state[4] = {
            theta_eq + a * v1_0 + b * v2_0,
            a * v1_1 + b * v2_1,
            a * v1_2 + b * v2_2,
            a * v1_3 + b * v2_3
        };

        double dt = -0.01;
        double cost = 0.0;
        bool valid = true;

        for(int step = 0; step < 1000; step++) {
            rk4_step_backward(state, dt, alpha, cost);
            if (isnan(state[0]) || isinf(state[0]) ||
                fabs(state[0]) > 1e8 || fabs(state[1]) > 1e8 ||
                fabs(state[2]) > 1e8 || fabs(state[3]) > 1e8) {
                valid = false;
                break;
            }
        }

        if (valid) {
            double res_sq = (state[0] - theta_target)*(state[0] - theta_target) +
                            (state[1] - phi_target)*(state[1] - phi_target);
            out_res[i] = res_sq;
            out_a[i] = a;
            out_b[i] = b;
            out_cost[i] = cost;
            out_l1[i] = state[2];
            out_l2[i] = state[3];
        } else {
            out_res[i] = DBL_MAX;
            out_cost[i] = DBL_MAX;
        }
    }
}

auto eval_trajectory_lambda = [](double a_val, double b_val, double theta_eq, double theta_target, double phi_target, double alpha,
                                 Eigen::Vector4d v1, Eigen::Vector4d v2, double& l1, double& l2) {
    double state[4] = {
        theta_eq + a_val * v1[0] + b_val * v2[0],
        a_val * v1[1] + b_val * v2[1],
        a_val * v1[2] + b_val * v2[2],
        a_val * v1[3] + b_val * v2[3]
    };
    double dt = -0.01;
    double cost = 0.0;
    for(int step = 0; step < 1000; step++) {
        rk4_step_backward(state, dt, alpha, cost);
        if (isnan(state[0]) || isinf(state[0]) ||
            fabs(state[0]) > 1e8 || fabs(state[1]) > 1e8 ||
            fabs(state[2]) > 1e8 || fabs(state[3]) > 1e8) {
            state[0] = NAN;
            state[1] = NAN;
            break;
        }
    }
    l1 = state[2];
    l2 = state[3];
    return std::make_pair(Eigen::Vector2d(state[0] - theta_target, state[1] - phi_target), cost);
};

void compute_residual_and_jacobian(double a, double b, double theta_eq, double theta_target, double phi_target, double alpha,
                                   Eigen::Vector4d v1, Eigen::Vector4d v2,
                                   Eigen::Vector2d& R, Eigen::Matrix2d& J, double& cost_val, double& l1_val, double& l2_val) {

    auto base_eval = eval_trajectory_lambda(a, b, theta_eq, theta_target, phi_target, alpha, v1, v2, l1_val, l2_val);
    R = base_eval.first;
    cost_val = base_eval.second;

    double eps = 1e-8;
    double dummy1, dummy2;
    Eigen::Vector2d R_da = eval_trajectory_lambda(a + eps, b, theta_eq, theta_target, phi_target, alpha, v1, v2, dummy1, dummy2).first;
    Eigen::Vector2d R_db = eval_trajectory_lambda(a, b + eps, theta_eq, theta_target, phi_target, alpha, v1, v2, dummy1, dummy2).first;

    J.col(0) = (R_da - R) / eps;
    J.col(1) = (R_db - R) / eps;
    
}

struct Candidate {
    double a, b, res_sq;
    bool operator<(const Candidate& other) const {
        return res_sq < other.res_sq;
    }
};

Result solve(double theta, double phi, double alpha) {
    Eigen::Matrix4d H;
    H << 0.0, 1.0, 0.0, 0.0,
         1.0, -alpha, 0.0, -1.0,
        -1.0, 0.0, 0.0, -1.0,
         0.0, -1.0, -1.0, alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(H);
    auto evals = es.eigenvalues();
    auto evecs = es.eigenvectors();

    Eigen::Vector4d v1, v2;
    int col = 0;
    for (int i = 0; i < 4; i++) {
        if (evals(i).real() < 0) {
            if (evals(i).imag() != 0.0) {
                v1 = evecs.col(i).real();
                v2 = evecs.col(i).imag();
                break;
            } else {
                if (col == 0) v1 = evecs.col(i).real();
                if (col == 1) v2 = evecs.col(i).real();
                col++;
                if (col == 2) break;
            }
        }
    }

    double best_global_cost = DBL_MAX;
    Result best_result = {0.0, 0.0, DBL_MAX};

    int k_offsets[] = {-1, 0, 1};
    for (int k : k_offsets) {
        double theta_eq = k * 2.0 * M_PI;

        std::vector<Candidate> all_candidates;

        int grid_pts = 256;
        int total_evals = grid_pts * grid_pts;
        double *d_a, *d_b, *d_res, *d_cost, *d_l1, *d_l2;
        cudaMalloc(&d_a, total_evals * sizeof(double));
        cudaMalloc(&d_b, total_evals * sizeof(double));
        cudaMalloc(&d_res, total_evals * sizeof(double));
        cudaMalloc(&d_cost, total_evals * sizeof(double));
        cudaMalloc(&d_l1, total_evals * sizeof(double));
        cudaMalloc(&d_l2, total_evals * sizeof(double));

        double *h_a = new double[total_evals];
        double *h_b = new double[total_evals];
        double *h_res = new double[total_evals];

        double radii[] = {0.1, 1.0, 10.0, 50.0};
        for(double r : radii) {
            coarse_search_kernel<<<BLOCKS, THREADS>>>(theta_eq, theta, phi, alpha,
                v1[0], v1[1], v1[2], v1[3],
                v2[0], v2[1], v2[2], v2[3],
                r, grid_pts, d_a, d_b, d_res, d_cost, d_l1, d_l2);
            cudaDeviceSynchronize();

            cudaMemcpy(h_a, d_a, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_b, d_b, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_res, d_res, total_evals * sizeof(double), cudaMemcpyDeviceToHost);

            for(int i = 0; i < total_evals; i++) {
                if(h_res[i] < 1e6) {
                    all_candidates.push_back({h_a[i], h_b[i], h_res[i]});
                }
            }
        }
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_res); cudaFree(d_cost); cudaFree(d_l1); cudaFree(d_l2);
        delete[] h_a; delete[] h_b; delete[] h_res;

        std::sort(all_candidates.begin(), all_candidates.end());

        int max_refine = std::min((int)all_candidates.size(), 3);

        for (int c = 0; c < max_refine; c++) {
            double a_opt = all_candidates[c].a;
            double b_opt = all_candidates[c].b;
            double final_cost = DBL_MAX, final_l1 = 0.0, final_l2 = 0.0;

            bool converged = false;
            for (int iter = 0; iter < 50; iter++) {
                Eigen::Vector2d R;
                Eigen::Matrix2d J;
                compute_residual_and_jacobian(a_opt, b_opt, theta_eq, theta, phi, alpha, v1, v2, R, J, final_cost, final_l1, final_l2);

                if (isnan(R.norm())) {
                    break;
                }

                if (R.norm() < 1e-6) {
                    converged = true;
                    break;
                }

                Eigen::Vector2d delta = -J.inverse() * R;
                if (isnan(delta.norm())) {
                    break;
                }

                double step_size = 1.0;
                double current_norm = R.norm();
                for(int ls = 0; ls < 5; ls++) {
                    double cand_a = a_opt + step_size * delta(0);
                    double cand_b = b_opt + step_size * delta(1);
                    double d1, d2;
                    Eigen::Vector2d cand_R = eval_trajectory_lambda(cand_a, cand_b, theta_eq, theta, phi, alpha, v1, v2, d1, d2).first;

                    if (!isnan(cand_R.norm()) && cand_R.norm() < current_norm) {
                        break;
                    }
                    step_size *= 0.5;
                }

                a_opt += step_size * delta(0);
                b_opt += step_size * delta(1);
            }

            if (converged && final_cost < best_global_cost) {
                best_result.l1 = final_l1;
                best_result.l2 = final_l2;
                best_result.cost = final_cost;
                best_global_cost = final_cost;
            }
        }
    }

    return best_result;
}
