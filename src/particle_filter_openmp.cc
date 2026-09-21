#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <array>
#include <string>
#include <random>
#include <cmath>

#include <omp.h>

int main(int argc, char* argv[])
{
    // ============================================================
    // Read simulated state-space data
    // ============================================================

    std::ifstream file("../results/state_space.csv");
    if(!file)
    {
        std::cerr << "File not found.\n";
        return 1;
    }
    std::vector<std::array<double, 5>> values;
    std::string line;
    std::getline(file, line);

    while(std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string t_str;
        std::string x_str;
        std::string y_str;

        std::getline(ss, t_str, ',');
        std::getline(ss, x_str, ',');
        std::getline(ss, y_str, ',');

        double t = std::stod(t_str);
        double x = std::stod(x_str);
        double y = std::stod(y_str);

        values.push_back({t, x, y});
    }

    if (values.empty())
    {
        std::cerr << "No observations found.\n";
        return 1;
    }

    // ============================================================
    // Model parameters
    // ============================================================

    constexpr double MU_X0{0.0};
    constexpr double SIGMA_X0{1.0};

    constexpr double A{0.9};
    constexpr double SIGMA_X{0.5};

    constexpr double C{1.0};
    constexpr double SIGMA_Y{1.0};

    std::size_t N_PARTICLES{100000};
    if (argc > 1)
    {
        N_PARTICLES = std::stoull(argv[1]);
    }
    constexpr int PF_SEED{12345};

    // time elapsed
    double time_propagation{0.0};
    double time_normalization{0.0};
    double time_mean{0.0};
    double time_resampling{0.0};

    // ============================================================
    // Particle arrays
    // ============================================================

    std::vector<double> particles(N_PARTICLES);
    std::vector<double> weights(N_PARTICLES);
    std::vector<double> resampled_particles(N_PARTICLES);

    std::vector<double> pf_mean(values.size());

    double total_start = omp_get_wtime();
    // ============================================================
    // t = 0
    //
    // 1. Sample particles from prior X_0 ~ N(mu_X0, sigma_X0^2)
    // 2. Weight particles using observation Y_0
    // 3. Normalize weights
    // 4. Calculate posterior PF mean
    // 5. Resample
    // ============================================================
    
    double Y_0 = values[0][2];
    double weight_sum{0.0};

    // create unique RNG with seed=PF_seed+i for thread i
    int n_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs(n_threads);
    for(int thread_id=0; thread_id<n_threads; ++thread_id)
    {
        std::seed_seq seed{PF_SEED, thread_id};
        rngs[thread_id].seed(seed);
    }

    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        auto& local_rng = rngs[thread_id];
        std::normal_distribution<double> initial_distribution(MU_X0, SIGMA_X0);
        
        #pragma omp for schedule(static)
        for(std::size_t i=0; i<N_PARTICLES; ++i)
        {
            // 1. Sample X_0^i using this thread's own RNG:
            particles[i] = initial_distribution(local_rng);

            // 2. MLE:
            double residual = Y_0 - C*particles[i];

            // 3. calc weights and add to thread's private reduction sum:
            weights[i] = std::exp(-0.5 * residual * residual / (SIGMA_Y * SIGMA_Y));
            weight_sum += weights[i];
        }
    }

    // weight_sum = 0.0;

    // #pragma omp parallel for schedule(static)
    // for (std::size_t i = 0; i < N_PARTICLES; ++i)
    // {
    //     weight_sum += weights[i];
    // }

    // double correct_weight_sum{0.0};

    // for (std::size_t i = 0; i < N_PARTICLES; ++i)
    // {
    //     correct_weight_sum += weights[i];
    // }

    // double critical_sum{0.0};

    // double start = omp_get_wtime();

    // #pragma omp parallel for schedule(static)
    // for (std::size_t i = 0; i < N_PARTICLES; ++i)
    // {
    //     #pragma omp critical
    //     {
    //         critical_sum += weights[i];
    //     }
    // }

    // double critical_time = omp_get_wtime() - start;

    // double reduction_sum{0.0};

    // double reduction_start = omp_get_wtime();

    // #pragma omp parallel for schedule(static) reduction(+:reduction_sum)
    // for (std::size_t i = 0; i < N_PARTICLES; ++i)
    // {
    //     reduction_sum += weights[i];
    // }

    // double reduction_time = omp_get_wtime() - reduction_start;

    // std::cout << "Reduction sum:  " << reduction_sum << '\n';
    // std::cout << "Correct sum:    " << correct_weight_sum << '\n';
    // std::cout << "Reduction time: " << reduction_time << " s\n";

    // std::cout << "Critical sum: " << critical_sum << '\n';
    // std::cout << "Correct sum:  " << correct_weight_sum << '\n';
    // std::cout << "Critical time: " << critical_time << " s\n";

    if (weight_sum == 0.0)
    {
        std::cerr << "Particle weights collapsed at t = 0.\n";
        return 1;
    }

    // Normalize weights
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N_PARTICLES; ++i)
    {
        weights[i] /= weight_sum;
    }

    // Posterior mean estimate E[X_0 | Y_0]
    double posterior_mean_0{0.0};
    #pragma omp parallel for schedule(static) reduction(+:posterior_mean_0)
    for (std::size_t i = 0; i < N_PARTICLES; ++i)
    {
        posterior_mean_0 += weights[i] * particles[i];
    }
    pf_mean[0] = posterior_mean_0;

    // Multinomial resampling
    std::discrete_distribution<std::size_t> resample_dist_0(weights.begin(), weights.end());

    std::mt19937 rng(PF_SEED);
    for (std::size_t i = 0; i < N_PARTICLES; ++i)
    {
        resampled_particles[i] = particles[resample_dist_0(rng)];
    }
    particles.swap(resampled_particles);

    // ============================================================
    // t = 1, ..., T-1
    // ============================================================
    for (std::size_t t = 1; t < values.size(); ++t)
    {
        double Y_t = values[t][2];
        weight_sum = 0.0;

        // --------------------------------------------------------
        // 1. Propagate particles
        // 2. Evaluate observation likelihood
        // --------------------------------------------------------

        double start = omp_get_wtime();
        #pragma omp parallel reduction(+:weight_sum)
        {
            int thread_id = omp_get_thread_num();
            auto& local_rng = rngs[thread_id];
            std::normal_distribution<double> local_standard_normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < N_PARTICLES; ++i)
            {
                double epsilon = local_standard_normal(local_rng);

                // propagate: X_t^(i) = A X_(t-1)^(i) + sigma_x epsilon_t^(i)
                particles[i] = A * particles[i] + SIGMA_X * epsilon;

                // obs residual:
                double residual = Y_t - C * particles[i];

                // unnormalized likelihood weight:
                weights[i] = std::exp(-0.5 * residual * residual / (SIGMA_Y * SIGMA_Y));

                // reduction:
                weight_sum += weights[i];
            }
        }
        time_propagation += omp_get_wtime() - start;

        // --------------------------------------------------------
        // 3. Normalize weights
        // --------------------------------------------------------

        if (weight_sum == 0.0)
        {
            std::cerr << "Particle weights collapsed at t = " << t << ".\n";
            return 1;
        }
        start = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            weights[i] /= weight_sum;
        }
        time_normalization += omp_get_wtime() - start;

        // --------------------------------------------------------
        // 4. Estimate posterior mean
        // --------------------------------------------------------

        start = omp_get_wtime();
        double pf_mean_t{0.0};
        #pragma omp parallel for schedule(static) reduction(+:pf_mean_t)
        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            pf_mean_t += weights[i] * particles[i];
        }
        pf_mean[t] = pf_mean_t;
        time_mean += omp_get_wtime() - start;

        // --------------------------------------------------------
        // 5. Multinomial resampling
        // --------------------------------------------------------

        start = omp_get_wtime();
        std::discrete_distribution<std::size_t> resample_dist(weights.begin(), weights.end());
        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            resampled_particles[i] = particles[resample_dist(rng)];
        }
        particles.swap(resampled_particles);
        time_resampling += omp_get_wtime() - start;
    }
    double total_runtime = omp_get_wtime() - total_start;



    std::cout << "Propagation + likelihood: " << time_propagation << " s\n";
    std::cout << "Normalization:            " << time_normalization << " s\n";
    std::cout << "Posterior mean:            " << time_mean << " s\n";
    std::cout << "Resampling:                " << time_resampling << " s\n";
    std::cout << "Runtime: " << total_runtime << " seconds\n";


    // ============================================================
    // Save results
    // ============================================================

    std::ofstream output("../results/particle_filter_openmp_N" + std::to_string(N_PARTICLES) + ".csv");

    if (!output)
    {
        std::cerr << "Could not open output file.\n";
        return 1;
    }

    output << "t,x_true,y_obs,pf_mean\n";
    for (std::size_t t = 0; t < values.size(); ++t)
    {
        output << t << ','
            << values[t][1] << ','
            << values[t][2] << ','
            << pf_mean[t] << '\n';
    }


    // ============================================================
    // RMSE against true hidden state
    // ============================================================

    double squared_error_sum{0.0};

    for (std::size_t t = 0; t < values.size(); ++t)
    {
        double error = pf_mean[t] - values[t][1];
        squared_error_sum += error * error;
    }
    double rmse = std::sqrt(squared_error_sum / values.size()
    );

    std::cout << "OpenMP Bootstrap Particle Filter\n";
    std::cout << "Particles: " << N_PARTICLES << '\n';
    std::cout << "PF RMSE:   " << rmse << '\n';
    std::cout << "Saved: ../results/particle_filter_omp_N" << N_PARTICLES << ".csv\n";

    return 0;
}