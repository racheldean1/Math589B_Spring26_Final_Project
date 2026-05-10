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
        if (isnan(y[i]) || isinf(y[i]) || fabs(y[i]) > 1.0e12) {
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

    double c = cos(theta);
    double s = sin(theta);

    f[0] = phi;
    f[1] = s - alpha * phi - l2 * c * c;
    f[2] = -s - l2 * c - l2 * l2 * s * c;
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

        /*
           Local stable patch near the equilibrium:
           y0 = a B1 + b B2
           where y0 = (theta, phi, lambda1, lambda2).
        */
        double y[4];
        y[0] = a * b1_0 + b * b2_0;
        y[1] = a * b1_1 + b * b2_1;
        y[2] = a * b1_2 + b * b2_2;
        y[3] = a * b1_3 + b * b2_3;

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

    y[0] = a * B1(0) + b * B2(0);
    y[1] = a * B1(1) + b * B2(1);
    y[2] = a * B1(2) + b * B2(2);
    y[3] = a * B1(3) + b * B2(3);

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
        theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        l1, l2
    );

    R = base.first;
    cost = base.second;

    double eps_a = 1.0e-7 * (1.0 + fabs(a));
    double eps_b = 1.0e-7 * (1.0 + fabs(b));

    double d1, d2;

    Eigen::Vector2d Ra_plus = eval_trajectory(
        a + eps_a, b,
        theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        d1, d2
    ).first;

    Eigen::Vector2d Ra_minus = eval_trajectory(
        a - eps_a, b,
        theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        d1, d2
    ).first;

    Eigen::Vector2d Rb_plus = eval_trajectory(
        a, b + eps_b,
        theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        d1, d2
    ).first;

    Eigen::Vector2d Rb_minus = eval_trajectory(
        a, b - eps_b,
        theta_eff, phi_target, alpha,
        B1, B2,
        nsteps, dt,
        d1, d2
    ).first;

    if (std::isfinite(Ra_plus.norm()) && std::isfinite(Ra_minus.norm())) {
        J.col(0) = (Ra_plus - Ra_minus) / (2.0 * eps_a);
    } else {
        J.col(0) = Eigen::Vector2d::Zero();
    }

    if (std::isfinite(Rb_plus.norm()) && std::isfinite(Rb_minus.norm())) {
        J.col(1) = (Rb_plus - Rb_minus) / (2.0 * eps_b);
    } else {
        J.col(1) = Eigen::Vector2d::Zero();
    }
}

Result fail_result() {
    Result r;
    r.l1 = 0.0;
    r.l2 = 0.0;
    r.cost = DBL_MAX;
    return r;
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

    /*
       Linearize the PMP system near the equilibrium.
       This gives the local stable eigenspace basis B = [B1 B2].
    */
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
            if (used == 0) {
                B1 = es.eigenvectors().col(i).real();
                used++;
            } else if (used == 1) {
                B2 = es.eigenvectors().col(i).real();
                used++;
                break;
            }
        }
    }

    if (B1.norm() == 0.0 || B2.norm() == 0.0) {
        return fail_result();
    }

    B1.normalize();
    B2 = B2 - B1.dot(B2) * B1;

    if (B2.norm() == 0.0) {
        return fail_result();
    }

    B2.normalize();

    int grid_pts = 256;
    int total_evals = grid_pts * grid_pts;

    double *d_a, *d_b, *d_res;

    if (cudaMalloc(&d_a, total_evals * sizeof(double)) != cudaSuccess) {
        return fail_result();
    }

    if (cudaMalloc(&d_b, total_evals * sizeof(double)) != cudaSuccess) {
        cudaFree(d_a);
        return fail_result();
    }

    if (cudaMalloc(&d_res, total_evals * sizeof(double)) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return fail_result();
    }

    double* h_a = new double[total_evals];
    double* h_b = new double[total_evals];
    double* h_res = new double[total_evals];

    std::vector<Candidate> all_candidates;

    /*
       Small radii matter because backward integration expands the stable directions.
       Large radii are included for the harder/high-energy cases.
    */
    double radii[] = {
        1.0e-9, 3.0e-9,
        1.0e-8, 3.0e-8,
        1.0e-7, 3.0e-7,
        1.0e-6, 3.0e-6,
        1.0e-5, 3.0e-5,
        1.0e-4, 3.0e-4,
        1.0e-3, 3.0e-3,
        1.0e-2, 3.0e-2,
        1.0e-1, 3.0e-1,
        1.0, 3.0, 10.0, 30.0
    };

    /*
       Longer horizons make the artificial starting point closer to equilibrium,
       which improves the cost and lambda accuracy.
    */
    double horizons[] = {4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};

    /*
       This follows the angle-well part of the algorithm directly:
       choose k near theta/(2*pi), then theta_eff = theta - 2*pi*k.
    */
    int k_center = (int) std::llround(theta / (2.0 * M_PI));

    for (double T : horizons) {
        int nsteps = (int) std::round(T / 0.005);
        double dt = -T / (double) nsteps;

        for (int dk = -2; dk <= 2; dk++) {
            int k = k_center + dk;
            double theta_eff = theta - 2.0 * M_PI * (double) k;

            for (double radius : radii) {
                coarse_search_kernel<<<BLOCKS, THREADS>>>(
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

                cudaError_t err = cudaDeviceSynchronize();

                if (err != cudaSuccess) {
                    continue;
                }

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
                        c.theta_eff = theta_eff;
                        c.T = T;
                        c.nsteps = nsteps;
                        local_candidates.push_back(c);
                    }
                }

                std::sort(local_candidates.begin(), local_candidates.end());

                int keep_per_launch = std::min((int) local_candidates.size(), 96);

                for (int j = 0; j < keep_per_launch; j++) {
                    all_candidates.push_back(local_candidates[j]);
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

    Result best = fail_result();

    std::sort(all_candidates.begin(), all_candidates.end());

    if (all_candidates.empty()) {
        return best;
    }

    double best_cost = DBL_MAX;
    double best_res = DBL_MAX;

    int max_refine = std::min((int) all_candidates.size(), 800);

    for (int c = 0; c < max_refine; c++) {
        double a = all_candidates[c].a;
        double b = all_candidates[c].b;
        double theta_eff = all_candidates[c].theta_eff;
        int nsteps = all_candidates[c].nsteps;
        double dt = -all_candidates[c].T / (double) nsteps;

        bool converged = false;

        double final_cost = DBL_MAX;
        double final_l1 = 0.0;
        double final_l2 = 0.0;
        double final_res = DBL_MAX;

        for (int iter = 0; iter < 80; iter++) {
            Eigen::Vector2d R;
            Eigen::Matrix2d J;

            compute_residual_and_jacobian(
                a, b,
                theta_eff, phi, alpha,
                B1, B2,
                nsteps, dt,
                R, J,
                final_cost, final_l1, final_l2
            );

            if (!std::isfinite(R.norm()) || !std::isfinite(final_cost)) {
                break;
            }

            final_res = R.norm();

            if (final_res < 1.0e-10) {
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

            for (int ls = 0; ls < 20; ls++) {
                double trial_a = a + step_size * delta(0);
                double trial_b = b + step_size * delta(1);

                double trial_l1, trial_l2;

                Eigen::Vector2d trial_R = eval_trajectory(
                    trial_a, trial_b,
                    theta_eff, phi, alpha,
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

        /*
           Prefer converged solutions with smallest cost.
           If nothing converges, keep the closest residual so we do not return
           DBL_MAX unless the search completely fails.
        */
        if (converged && final_cost < best_cost) {
            best.l1 = final_l1;
            best.l2 = final_l2;
            best.cost = final_cost;
            best_cost = final_cost;
            best_res = final_res;
        } else if (best_cost == DBL_MAX && final_res < best_res && std::isfinite(final_cost)) {
            best.l1 = final_l1;
            best.l2 = final_l2;
            best.cost = final_cost;
            best_res = final_res;
        }
    }

    return best;
}
