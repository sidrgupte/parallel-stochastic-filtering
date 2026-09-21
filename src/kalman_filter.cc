#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <array>
#include <string>

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
    
    constexpr double MU_X0{0.0f};
    constexpr double SIGMA_X0{1.0f};
    constexpr double A{0.9f};
    constexpr double SIGMA_X{0.5f};

    constexpr double C{1.0f};
    constexpr double SIGMA_Y{1.0f};

    double Q = SIGMA_X*SIGMA_X;
    double R = SIGMA_Y*SIGMA_Y;

    // predicted mean/variance PRE observing Y_0 (PRIOR):
    std::vector<double> m_pre(values.size());
    m_pre[0] = MU_X0; 

    std::vector<double> p_pre(values.size());
    p_pre[0] = SIGMA_X0*SIGMA_X0;

    // update values after seeing Y_0:
    double Y_0 = values[0][2];
    std::vector<double> innovation(values.size()), innovation_variance(values.size());
    std::vector<double> kalman_gain(values.size());
    innovation[0] = Y_0 - C*m_pre[0];
    innovation_variance[0] = (C*C)*p_pre[0] +  R;
    kalman_gain[0] = (p_pre[0]*C) / innovation_variance[0];    


    // predicted mean/variance POST observing Y_0 (POSTERIOR):
    std::vector<double> m_post(values.size());
    m_post[0] = m_pre[0] + kalman_gain[0]*innovation[0]; 

    std::vector<double> p_post(values.size());
    p_post[0] = (1-kalman_gain[0]*C)*p_pre[0];

    // loop over the values:
    for(size_t t=1; t<values.size(); ++t)
    {

        // 1. Predict:
        m_pre[t] = A*m_post[t-1];
        p_pre[t] = (A*A)*p_post[t-1] + Q;

        // 2. Look at Y_t:
        double Y_t = values[t][2];
        innovation[t] = Y_t - C*m_pre[t];
        innovation_variance[t] = (C*C)*p_pre[t] + R;
        kalman_gain[t] = (p_pre[t]*C) / innovation_variance[t];

        // 3. Update:
        m_post[t] = m_pre[t] + kalman_gain[t]*innovation[t];
        p_post[t] = (1-kalman_gain[t]*C) * p_pre[t];
    }

    // save:
    std::ofstream kf_file("../results/kalman_filter.csv");
    kf_file << "t,x_true,y_obs,m_post\n";
    if (!kf_file)
    {
        std::cerr << "Could not open kalman_filter.csv.\n";
        return 1;
    }

    for(size_t t = 0; t<values.size(); ++t)
    {
        kf_file << t << "," << values[t][1] << "," << values[t][2] << "," << m_post[t] << "\n";
    }
    std::cout << "Kalman Filter files saved successfully!" << std::endl;

    return 0;
}