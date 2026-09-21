#include <iostream>
#include <random>
#include <vector>
#include <fstream>

double sample_normal(const double mu, const double sigma, std::mt19937& rng)
{
    std::normal_distribution<double> normal(mu, sigma);
    return normal(rng);
}

int main()
{
    std::cout << "Parallel Stochastic Filtering\n";

    // CONSTANTS:
    constexpr int SEED{42};
    constexpr int T{500};

    constexpr double MU_X0{0.0f};
    constexpr double SIGMA_X0{1.0f};
    constexpr double A{0.9f};
    constexpr double SIGMA_X{0.5f};
    constexpr double MU_EPSILON{0.0f};
    constexpr double SIGMA_EPSILON{1.0f};

    constexpr double C{1.0f};
    constexpr double SIGMA_Y{1.0f};
    constexpr double MU_ETA{0.0f};
    constexpr double SIGMA_ETA{1.0f};

    // seed the generator:
    std::mt19937 rng(SEED);

    // Initial State: 
    // X(t+1) = aX(t) + sigma_x*epsilon_t
    std::vector<double>X(T);
    X[0] = sample_normal(MU_X0, SIGMA_X0, rng);

    // Initial Observation: 
    // Y(t) = cX + sigma_y*eta_t
    std::vector<double>Y(T);
    double eta_0 = sample_normal(MU_ETA, SIGMA_ETA, rng);
    Y[0] = C*X[0] + SIGMA_Y*eta_0;

    // populate (X_t, Y_t):
    for(size_t t=1; t<T; ++t)
    {
        double epsilon_t = sample_normal(MU_EPSILON, SIGMA_EPSILON, rng);
        double eta_t = sample_normal(MU_ETA, SIGMA_ETA, rng);

        X[t] = A*X[t-1] + SIGMA_X*epsilon_t;
        Y[t] = C*X[t] + SIGMA_Y*eta_t;

    }

    // write the vals:
    std::ofstream file("../results/state_space.csv");
    if(!file)
    {
        std::cerr << "Error: could not open the output file.\n";
        return 1;
    }
    file << "t, x_true, y_obs\n";
    for(size_t t=0; t<T; ++t)
    {
        file << t << "," << X[t] << "," << Y[t] << "\n";
    }

    return 0;
}