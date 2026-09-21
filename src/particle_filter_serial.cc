#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <array>
#include <string>
#include <random>
#include <cmath>

int main()
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

    constexpr std::size_t N_PARTICLES{500000};
    constexpr int PF_SEED{12345};

    std::mt19937 rng(PF_SEED);
    std::normal_distribution<double> initial_distribution(MU_X0, SIGMA_X0);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    // ============================================================
    // Particle arrays
    // ============================================================

    std::vector<double> particles(N_PARTICLES);
    std::vector<double> weights(N_PARTICLES);
    std::vector<double> resampled_particles(N_PARTICLES);

    std::vector<double> pf_mean(values.size());


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

    for (std::size_t i = 0; i < N_PARTICLES; ++i)
    {
        // Sample X_0^(i)
        particles[i] = initial_distribution(rng);

        // Observation residual:
        // Y_0 - C X_0^(i)
        double residual = Y_0 - C * particles[i];

        // Gaussian observation likelihood.
        // Constant factor is unnecessary because weights are normalized.
        weights[i] = std::exp(-0.5 * residual * residual / (SIGMA_Y * SIGMA_Y));
        weight_sum += weights[i];
    }

    if (weight_sum == 0.0)
    {
        std::cerr << "Particle weights collapsed at t = 0.\n";
        return 1;
    }

    // Normalize weights
    for (std::size_t i = 0; i < N_PARTICLES; ++i)
    {
        weights[i] /= weight_sum;
    }

    // Posterior mean estimate E[X_0 | Y_0]
    pf_mean[0] = 0.0;

    for (std::size_t i = 0; i < N_PARTICLES; ++i)
    {
        pf_mean[0] += weights[i] * particles[i];
    }

    // Multinomial resampling
    std::discrete_distribution<std::size_t> resample_dist_0(weights.begin(), weights.end());

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

        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            double epsilon = standard_normal(rng);

            // X_t^(i) = A X_(t-1)^(i) + sigma_x epsilon_t^(i)
            particles[i] = A * particles[i] + SIGMA_X * epsilon;

            // Observation residual
            double residual = Y_t - C * particles[i];

            // Unnormalized likelihood weight
            weights[i] = std::exp(-0.5 * residual * residual / (SIGMA_Y * SIGMA_Y));
            weight_sum += weights[i];
        }


        // --------------------------------------------------------
        // 3. Normalize weights
        // --------------------------------------------------------

        if (weight_sum == 0.0)
        {
            std::cerr << "Particle weights collapsed at t = " << t << ".\n";
            return 1;
        }

        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            weights[i] /= weight_sum;
        }


        // --------------------------------------------------------
        // 4. Estimate posterior mean
        // --------------------------------------------------------

        pf_mean[t] = 0.0;

        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            pf_mean[t] += weights[i] * particles[i];
        }


        // --------------------------------------------------------
        // 5. Multinomial resampling
        // --------------------------------------------------------

        std::discrete_distribution<std::size_t> resample_dist(weights.begin(), weights.end());
        for (std::size_t i = 0; i < N_PARTICLES; ++i)
        {
            resampled_particles[i] = particles[resample_dist(rng)];
        }

        particles.swap(resampled_particles);
    }


    // ============================================================
    // Save results
    // ============================================================

    std::ofstream output("../results/particle_filter_serial_N" + std::to_string(N_PARTICLES) + ".csv");

    if (!output)
    {
        std::cerr << "Could not open output file.\n";
        return 1;
    }

    output << "t,x_true,y_obs,pf_mean\n";

    for (std::size_t t = 0; t < values.size(); ++t)
    {
        output
            << t << ','
            << values[t][1] << ','
            << values[t][2] << ','
            << pf_mean[t]
            << '\n';
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

    std::cout << "Serial Bootstrap Particle Filter\n";
    std::cout << "Particles: " << N_PARTICLES << '\n';
    std::cout << "PF RMSE:   " << rmse << '\n';
    std::cout << "Saved: ../results/particle_filter_serial.csv\n";

    return 0;
}
   
