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

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#define THREADS 256
#define BLOCKS 256

struct Candidate {
    double a;
    double b;
    double res_sq;
    double theta_eq;
    double theta_eff;
    double T;
    int nsteps;

    bool operator<(const Candidate& other) const {
        return res_sq < other.res_sq;
    }
};

__device__ __host__
bool finite_state(const double* y) {
    for (int i = 0; i < 4; i++) {
        if (isnan(y[i]) || isinf(y[i]) || fabs(y[i]) > 1.0e10) {
            return false;
        }
    }
    return true;
}

__device__ __host__
void system_dynamics(const double* y, double* f, double alpha) {
    double theta = y[0];
    double phi   = y[1];
    double l1    = y[2];
    double l2    = y[3];

    f[0] = phi;
    f[1] = sin(theta) - alpha * phi - l2 * cos(theta) * cos(theta);
    f[2] = -sin(theta) - l2 * cos(theta) - l2 * l2 * sin(theta) * cos(theta);
    f[3] = -phi - l1 + alpha * l2;
}

__device__ __host__
double running_cost(const double* y) {
    double theta = y[0];
    double phi   = y[1];
    double l2    = y[3];

    double u = -l2 * cos(theta);

    return (1.0 - cos(theta)) + 0.5 * phi * phi + 0.5 * u * u;
}

__device__ __host__
void rk4_step(double* y, double dt, double alpha, double& cost) {
    double k1[4], k2[4], k3[4], k4[4], tmp[4];

    system_dynamics(y, k1, alpha);
    double c1 = running_cost(y);

    for (int i = 0; i < 4; i++) {
        tmp[i] = y[i] + 0.5 * dt * k1[i];
    }
    system_dynamics(tmp, k2, alpha);
    double c2 = running_cost(tmp);

    for (int i = 0; i < 4; i++) {
        tmp[i] = y[i] + 0.5 * dt * k2[i];
    }
    system_dynamics(tmp, k3, alpha);
    double c3 = running_cost(tmp);

    for (int i = 0; i < 4; i++) {
        tmp[i] = y[i] + dt * k3[i];
    }
    system_dynamics(tmp, k4, alpha);
    double c4 = running_cost(tmp);

    for (int i = 0; i < 4; i++) {
        y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    cost += (fabs(dt) / 6.0) * (c1 + 2.0 * c2 + 2.0 * c3 + c4);
}

__global__
void coarse_search_kernel(
    double theta_eq,
    double theta_eff,
    double phi_target,
    double alpha,
    double b1_0, double b1_1, double b1_2, double b1_3,
    double b2_0, double b2_1, double b2_2, double b2_3,
    double radius,
    int grid_pts,
    int nsteps,
    double dt,
    double* out_a,
    double* out_b,
    double* out_res
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int total = grid_pts * grid_pts;

    for (int idx = tid; idx < total; idx += total_threads) {
        int row = idx / grid_pts;
        int col = idx % grid_pts;

        double a = -radius + 2.0 * radius * ((double) row / (double) (grid_pts - 1));
        double b = -radius + 2.0 * radius * ((double) col / (double) (grid_pts - 1));

        double y[4];

        y[0] = theta_eq + a * b1_0 + b * b2_0;
        y[1] =          a * b1_1 + b * b2_1;
        y[2] =          a * b1_2 + b * b2_2;
        y[3] =          a * b1_3 + b * b2_3;

        double cost = 0.0;
        bool valid = true;

        for (int step = 0; step < nsteps; step++) {
            rk4_step(y, dt, alpha, cost);

            if (!finite_state(y)) {
                valid = false;
                break;
            }
        }

        out_a[idx] = a;
        out_b[idx] = b;

        if (valid) {
            double r0 = y[0] - theta_eff;
            double r1 = y[1] - phi_target;
            out_res[idx] = r0 * r0 + r1 * r1;
        } else {
            out_res[idx] = DBL_MAX;
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
    const Eigen::Vector4d& B1,
    const Eigen::Vector4d& B2,
    int nsteps,
    double dt,
    double& l1_end,
    double& l2_end
) {
    double y[4];

    y[0] = theta_eq + a * B1(0) + b * B2(0);
    y[1] =          a * B1(1) + b * B2(1);
    y[2] =          a * B1(2) + b * B2(2);
    y[3] =          a * B1(3) + b * B2(3);

    double cost = 0.0;

    for (int step = 0; step < nsteps; step++) {
        rk4_step(y, dt, alpha, cost);

        if (!finite_state(y)) {
            l1_end = NAN;
            l2_end = NAN;
            return std::make_pair(Eigen::Vector2d(NAN, NAN), DBL_MAX);
        }
    }

    l1_end = y[2];
    l2_end = y[3];

    Eigen::Vector2d R;
    R(0) = y[0] - theta_eff;
    R(1) = y[1] - phi_target;

    return std::make_pair(R, cost);
}

void compute_residual_and_jacobian(
    double a,
    double b,
    double theta_eq,
    double theta_eff,
    double phi_target,
    double alpha,
    const Eigen::Vector4d& B1,
    const Eigen::Vector4d& B2,
    int nsteps,
    double dt,
    Eigen::Vector2d& R,
    Eigen::Matrix2d& J,
    double& cost,
    double& l1,
    double& l2
) {
    auto base = eval_trajectory(
        a, b,
        theta_eq, theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        l1, l2
    );

    R = base.first;
    cost = base.second;

    double eps = 1.0e-6 * (1.0 + fabs(a) + fabs(b));

    double d1, d2;

    Eigen::Vector2d Ra = eval_trajectory(
        a + eps, b,
        theta_eq, theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        d1, d2
    ).first;

    Eigen::Vector2d Rb = eval_trajectory(
        a, b + eps,
        theta_eq, theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        d1, d2
    ).first;

    J.col(0) = (Ra - R) / eps;
    J.col(1) = (Rb - R) / eps;
}

Result solve(double theta, double phi, double alpha) {
    double theta_mod = std::remainder(theta, 2.0 * M_PI);

    if (fabs(theta_mod) < 1.0e-13 && fabs(phi) < 1.0e-13) {
        Result exact;
        exact.l1 = 0.0;
        exact.l2 = 0.0;
        exact.cost = 0.0;
        return exact;
    }

    Eigen::Matrix4d A;

    A <<  0.0,  1.0,     0.0,  0.0,
          1.0, -alpha,   0.0, -1.0,
         -1.0,  0.0,     0.0, -1.0,
          0.0, -1.0,    -1.0,  alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(A);

    Eigen::Vector4d B1 = Eigen::Vector4d::Zero();
    Eigen::Vector4d B2 = Eigen::Vector4d::Zero();

    int used = 0;

    for (int i = 0; i < 4; i++) {
        std::complex<double> ev = es.eigenvalues()(i);

        if (ev.real() < 0.0) {
            if (fabs(ev.imag()) > 1.0e-12) {
                B1 = es.eigenvectors().col(i).real();
                B2 = es.eigenvectors().col(i).imag();
                used = 2;
                break;
            } else {
                if (used == 0) {
                    B1 = es.eigenvectors().col(i).real();
                } else if (used == 1) {
                    B2 = es.eigenvectors().col(i).real();
                }

                used++;

                if (used == 2) {
                    break;
                }
            }
        }
    }

    if (B1.norm() == 0.0 || B2.norm() == 0.0) {
        Result fail;
        fail.l1 = 0.0;
        fail.l2 = 0.0;
        fail.cost = DBL_MAX;
        return fail;
    }

    B1.normalize();

    B2 = B2 - B1.dot(B2) * B1;

    if (B2.norm() == 0.0) {
        Result fail;
        fail.l1 = 0.0;
        fail.l2 = 0.0;
        fail.cost = DBL_MAX;
        return fail;
    }

    B2.normalize();

    int grid_pts = 256;
    int total_evals = grid_pts * grid_pts;

    double *d_a, *d_b, *d_res;

    cudaMalloc(&d_a, total_evals * sizeof(double));
    cudaMalloc(&d_b, total_evals * sizeof(double));
    cudaMalloc(&d_res, total_evals * sizeof(double));

    double* h_a = new double[total_evals];
    double* h_b = new double[total_evals];
    double* h_res = new double[total_evals];

    std::vector<Candidate> all_candidates;

    double radii[] = {
        1.0e-8, 3.0e-8,
        1.0e-7, 3.0e-7,
        1.0e-6, 3.0e-6,
        1.0e-5, 3.0e-5,
        1.0e-4, 3.0e-4,
        1.0e-3, 3.0e-3,
        1.0e-2, 3.0e-2,
        1.0e-1, 3.0e-1,
        1.0, 3.0, 10.0
    };

    double horizons[] = {6.0, 8.0, 10.0, 12.0, 14.0};

    for (double T : horizons) {
        int nsteps = (int) std::round(T / 0.005);
        double dt = -T / (double) nsteps;

        for (int target_shift = -2; target_shift <= 2; target_shift++) {
            double theta_eff = theta_mod + 2.0 * M_PI * target_shift;

            for (int eq_shift = -2; eq_shift <= 2; eq_shift++) {
                double theta_eq = 2.0 * M_PI * eq_shift;

                for (double radius : radii) {
                    coarse_search_kernel<<<BLOCKS, THREADS>>>(
                        theta_eq,
                        theta_eff,
                        phi,
                        alpha,
                        B1(0), B1(1), B1(2), B1(3),
                        B2(0), B2(1), B2(2), B2(3),
                        radius,
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

                    std::vector<Candidate> local_candidates;

                    for (int i = 0; i < total_evals; i++) {
                        if (std::isfinite(h_res[i]) && h_res[i] < DBL_MAX) {
                            Candidate c;
                            c.a = h_a[i];
                            c.b = h_b[i];
                            c.res_sq = h_res[i];
                            c.theta_eq = theta_eq;
                            c.theta_eff = theta_eff;
                            c.T = T;
                            c.nsteps = nsteps;
                            local_candidates.push_back(c);
                        }
                    }

                    std::sort(local_candidates.begin(), local_candidates.end());

                    int keep_per_gpu_launch = std::min((int) local_candidates.size(), 64);

                    for (int j = 0; j < keep_per_gpu_launch; j++) {
                        all_candidates.push_back(local_candidates[j]);
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

    Result best;
    best.l1 = 0.0;
    best.l2 = 0.0;
    best.cost = DBL_MAX;

    std::sort(all_candidates.begin(), all_candidates.end());

    if (all_candidates.empty()) {
        return best;
    }

    double best_cost = DBL_MAX;

    int max_refine = std::min((int) all_candidates.size(), 500);

    for (int c = 0; c < max_refine; c++) {
        double a = all_candidates[c].a;
        double b = all_candidates[c].b;

        double theta_eq = all_candidates[c].theta_eq;
        double theta_eff = all_candidates[c].theta_eff;
        int nsteps = all_candidates[c].nsteps;
        double dt = -all_candidates[c].T / (double) nsteps;

        bool converged = false;

        double final_cost = DBL_MAX;
        double final_l1 = 0.0;
        double final_l2 = 0.0;

        for (int iter = 0; iter < 60; iter++) {
            Eigen::Vector2d R;
            Eigen::Matrix2d J;

            compute_residual_and_jacobian(
                a, b,
                theta_eq, theta_eff, phi, alpha,
                B1, B2,
                nsteps, dt,
                R, J,
                final_cost, final_l1, final_l2
            );

            if (!std::isfinite(R.norm()) || !std::isfinite(final_cost)) {
                break;
            }

            if (R.norm() < 1.0e-8) {
                converged = true;
                break;
            }

            Eigen::Vector2d delta = J.fullPivLu().solve(-R);

            if (!std::isfinite(delta.norm())) {
                break;
            }

            double current_norm = R.norm();
            double step_size = 1.0;
            bool accepted = false;

            for (int ls = 0; ls < 15; ls++) {
                double trial_a = a + step_size * delta(0);
                double trial_b = b + step_size * delta(1);

                double trial_l1, trial_l2;

                Eigen::Vector2d trial_R = eval_trajectory(
                    trial_a, trial_b,
                    theta_eq, theta_eff, phi, alpha,
                    B1, B2,
                    nsteps, dt,
                    trial_l1, trial_l2
                ).first;

                if (std::isfinite(trial_R.norm()) && trial_R.norm() < current_norm) {
                    accepted = true;
                    break;
                }

                step_size *= 0.5;
            }

            if (!accepted) {
                break;
            }

            a += step_size * delta(0);
            b += step_size * delta(1);
        }

        if (converged && final_cost < best_cost) {
            best.l1 = final_l1;
            best.l2 = final_l2;
            best.cost = final_cost;
            best_cost = final_cost;
        }
    }

    return best;
}
