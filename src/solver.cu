#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <utility>

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "solver.hpp"

#define THREADS 256
#define BLOCKS 256

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Candidate {
    double a;
    double b;
    double res_sq;
    double theta_eq;
    double theta_eff;

    bool operator<(const Candidate& other) const {
        return res_sq < other.res_sq;
    }
};

__device__ __host__
void system_dynamics(const double* state, double* deriv, double alpha) {
    double theta = state[0];
    double phi   = state[1];
    double l1    = state[2];
    double l2    = state[3];

    deriv[0] = phi;
    deriv[1] = sin(theta) - alpha * phi - l2 * cos(theta) * cos(theta);
    deriv[2] = -sin(theta) - l2 * cos(theta) - l2 * l2 * sin(theta) * cos(theta);
    deriv[3] = -phi - l1 + alpha * l2;
}

__device__ __host__
double lagrangian_cost(const double* state) {
    double theta = state[0];
    double phi   = state[1];
    double l2    = state[3];

    double u = -l2 * cos(theta);

    return (1.0 - cos(theta)) + 0.5 * phi * phi + 0.5 * u * u;
}

__device__ __host__
bool bad_state(const double* state) {
    for (int i = 0; i < 4; i++) {
        if (isnan(state[i]) || isinf(state[i]) || fabs(state[i]) > 1.0e9) {
            return true;
        }
    }
    return false;
}

__device__ __host__
void rk4_step_backward(double* state, double dt, double alpha, double& cost_accum) {
    double k1[4], k2[4], k3[4], k4[4], temp[4];

    system_dynamics(state, k1, alpha);
    double c1 = lagrangian_cost(state);

    for (int i = 0; i < 4; i++) {
        temp[i] = state[i] + 0.5 * dt * k1[i];
    }
    system_dynamics(temp, k2, alpha);
    double c2 = lagrangian_cost(temp);

    for (int i = 0; i < 4; i++) {
        temp[i] = state[i] + 0.5 * dt * k2[i];
    }
    system_dynamics(temp, k3, alpha);
    double c3 = lagrangian_cost(temp);

    for (int i = 0; i < 4; i++) {
        temp[i] = state[i] + dt * k3[i];
    }
    system_dynamics(temp, k4, alpha);
    double c4 = lagrangian_cost(temp);

    for (int i = 0; i < 4; i++) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    cost_accum += (fabs(dt) / 6.0) * (c1 + 2.0 * c2 + 2.0 * c3 + c4);
}

__global__
void coarse_search_kernel(
    double theta_eq,
    double theta_eff,
    double phi_target,
    double alpha,
    double v1_0, double v1_1, double v1_2, double v1_3,
    double v2_0, double v2_1, double v2_2, double v2_3,
    double r,
    int grid_pts,
    int nsteps,
    double dt,
    double* out_a,
    double* out_b,
    double* out_res
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total = grid_pts * grid_pts;

    for (int i = tid; i < total; i += total_threads) {
        int row = i / grid_pts;
        int col = i % grid_pts;

        double a = -r + 2.0 * r * ((double) row / (double) (grid_pts - 1));
        double b = -r + 2.0 * r * ((double) col / (double) (grid_pts - 1));

        double state[4] = {
            theta_eq + a * v1_0 + b * v2_0,
            0.0      + a * v1_1 + b * v2_1,
            0.0      + a * v1_2 + b * v2_2,
            0.0      + a * v1_3 + b * v2_3
        };

        double cost = 0.0;
        bool valid = true;

        for (int step = 0; step < nsteps; step++) {
            rk4_step_backward(state, dt, alpha, cost);

            if (bad_state(state)) {
                valid = false;
                break;
            }
        }

        if (valid) {
            double dtheta = state[0] - theta_eff;
            double dphi   = state[1] - phi_target;

            out_a[i] = a;
            out_b[i] = b;
            out_res[i] = dtheta * dtheta + dphi * dphi;
        } else {
            out_a[i] = a;
            out_b[i] = b;
            out_res[i] = DBL_MAX;
        }
    }
}

std::pair<Eigen::Vector2d, double>
eval_trajectory(
    double a,
    double b,
    double theta_eq,
    double theta_eff,
    double phi_target,
    double alpha,
    const Eigen::Vector4d& v1,
    const Eigen::Vector4d& v2,
    int nsteps,
    double dt,
    double& l1,
    double& l2
) {
    double state[4] = {
        theta_eq + a * v1[0] + b * v2[0],
        0.0      + a * v1[1] + b * v2[1],
        0.0      + a * v1[2] + b * v2[2],
        0.0      + a * v1[3] + b * v2[3]
    };

    double cost = 0.0;

    for (int step = 0; step < nsteps; step++) {
        rk4_step_backward(state, dt, alpha, cost);

        if (bad_state(state)) {
            l1 = NAN;
            l2 = NAN;
            return {Eigen::Vector2d(NAN, NAN), DBL_MAX};
        }
    }

    l1 = state[2];
    l2 = state[3];

    Eigen::Vector2d R;
    R(0) = state[0] - theta_eff;
    R(1) = state[1] - phi_target;

    return {R, cost};
}

void compute_residual_and_jacobian(
    double a,
    double b,
    double theta_eq,
    double theta_eff,
    double phi_target,
    double alpha,
    const Eigen::Vector4d& v1,
    const Eigen::Vector4d& v2,
    int nsteps,
    double dt,
    Eigen::Vector2d& R,
    Eigen::Matrix2d& J,
    double& cost_val,
    double& l1_val,
    double& l2_val
) {
    auto base_eval = eval_trajectory(
        a, b, theta_eq, theta_eff, phi_target, alpha,
        v1, v2, nsteps, dt, l1_val, l2_val
    );

    R = base_eval.first;
    cost_val = base_eval.second;

    double eps = 1.0e-6 * (1.0 + fabs(a) + fabs(b));

    double l1_dummy, l2_dummy;

    Eigen::Vector2d R_da = eval_trajectory(
        a + eps, b, theta_eq, theta_eff, phi_target, alpha,
        v1, v2, nsteps, dt, l1_dummy, l2_dummy
    ).first;

    Eigen::Vector2d R_db = eval_trajectory(
        a, b + eps, theta_eq, theta_eff, phi_target, alpha,
        v1, v2, nsteps, dt, l1_dummy, l2_dummy
    ).first;

    J.col(0) = (R_da - R) / eps;
    J.col(1) = (R_db - R) / eps;
}

Result solve(double theta, double phi, double alpha) {
    Eigen::Matrix4d H;

    H << 0.0,  1.0,    0.0,  0.0,
         1.0, -alpha, 0.0, -1.0,
        -1.0,  0.0,    0.0, -1.0,
         0.0, -1.0,   -1.0,  alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(H);

    Eigen::Vector4d v1 = Eigen::Vector4d::Zero();
    Eigen::Vector4d v2 = Eigen::Vector4d::Zero();

    int col = 0;

    for (int i = 0; i < 4; i++) {
        std::complex<double> eval = es.eigenvalues()(i);

        if (eval.real() < 0.0) {
            if (fabs(eval.imag()) > 1.0e-12) {
                v1 = es.eigenvectors().col(i).real();
                v2 = es.eigenvectors().col(i).imag();
                break;
            } else {
                if (col == 0) {
                    v1 = es.eigenvectors().col(i).real();
                } else if (col == 1) {
                    v2 = es.eigenvectors().col(i).real();
                }

                col++;

                if (col == 2) {
                    break;
                }
            }
        }
    }

    v1.normalize();
    v2.normalize();

    double theta_mod = std::remainder(theta, 2.0 * M_PI);

    double best_global_cost = DBL_MAX;
    Result best_result;
    best_result.l1 = 0.0;
    best_result.l2 = 0.0;
    best_result.cost = DBL_MAX;

    int grid_pts = 256;
    int total_evals = grid_pts * grid_pts;

    int nsteps = 3000;
    double dt = -0.01;

    double* d_a;
    double* d_b;
    double* d_res;

    cudaMalloc(&d_a, total_evals * sizeof(double));
    cudaMalloc(&d_b, total_evals * sizeof(double));
    cudaMalloc(&d_res, total_evals * sizeof(double));

    double* h_a = new double[total_evals];
    double* h_b = new double[total_evals];
    double* h_res = new double[total_evals];

    std::vector<Candidate> all_candidates;

    double radii[] = {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0};

    for (int target_shift = -2; target_shift <= 2; target_shift++) {
        double theta_eff = theta_mod + 2.0 * M_PI * target_shift;

        for (int eq_shift = -2; eq_shift <= 2; eq_shift++) {
            double theta_eq = 2.0 * M_PI * eq_shift;

            for (double r : radii) {
                coarse_search_kernel<<<BLOCKS, THREADS>>>(
                    theta_eq,
                    theta_eff,
                    phi,
                    alpha,
                    v1[0], v1[1], v1[2], v1[3],
                    v2[0], v2[1], v2[2], v2[3],
                    r,
                    grid_pts,
                    nsteps,
                    dt,
                    d_a,
                    d_b,
                    d_res
                );

                cudaDeviceSynchronize();

                cudaMemcpy(h_a, d_a, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_b, d_b, total_evals * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_res, d_res, total_evals * sizeof(double), cudaMemcpyDeviceToHost);

                for (int i = 0; i < total_evals; i++) {
                    if (h_res[i] < 1.0e4) {
                        Candidate cand;
                        cand.a = h_a[i];
                        cand.b = h_b[i];
                        cand.res_sq = h_res[i];
                        cand.theta_eq = theta_eq;
                        cand.theta_eff = theta_eff;
                        all_candidates.push_back(cand);
                    }
                }
            }
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    delete[] h_a;
    delete[] h_b;
    delete[] h_res;

    std::sort(all_candidates.begin(), all_candidates.end());

    int max_refine = std::min((int) all_candidates.size(), 50);

    for (int c = 0; c < max_refine; c++) {
        double a_opt = all_candidates[c].a;
        double b_opt = all_candidates[c].b;

        double theta_eq = all_candidates[c].theta_eq;
        double theta_eff = all_candidates[c].theta_eff;

        double final_cost = DBL_MAX;
        double final_l1 = 0.0;
        double final_l2 = 0.0;

        bool converged = false;

        for (int iter = 0; iter < 60; iter++) {
            Eigen::Vector2d R;
            Eigen::Matrix2d J;

            compute_residual_and_jacobian(
                a_opt, b_opt,
                theta_eq, theta_eff, phi, alpha,
                v1, v2,
                nsteps, dt,
                R, J,
                final_cost, final_l1, final_l2
            );

            if (!std::isfinite(R.norm())) {
                break;
            }

            if (R.norm() < 1.0e-7) {
                converged = true;
                break;
            }

            Eigen::Vector2d delta = J.fullPivLu().solve(-R);

            if (!std::isfinite(delta.norm())) {
                break;
            }

            double step_size = 1.0;
            double current_norm = R.norm();

            bool accepted = false;

            for (int ls = 0; ls < 10; ls++) {
                double cand_a = a_opt + step_size * delta(0);
                double cand_b = b_opt + step_size * delta(1);

                double tmp_l1, tmp_l2;

                Eigen::Vector2d cand_R = eval_trajectory(
                    cand_a, cand_b,
                    theta_eq, theta_eff, phi, alpha,
                    v1, v2,
                    nsteps, dt,
                    tmp_l1, tmp_l2
                ).first;

                if (std::isfinite(cand_R.norm()) && cand_R.norm() < current_norm) {
                    accepted = true;
                    break;
                }

                step_size *= 0.5;
            }

            if (!accepted) {
                break;
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

    return best_result;
}
