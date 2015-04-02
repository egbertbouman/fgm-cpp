#include <map>  
#include <string>
#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd double_matrix(Eigen::MatrixXd& m)
{
    m *= 2;
    return m;
}

Eigen::MatrixXd fgm(Eigen::MatrixXd& KP, Eigen::MatrixXd& KQ, Eigen::MatrixXd& Ct, Eigen::MatrixXd& asgTX,
                    std::map<std::string, Eigen::MatrixXd>& gph1, std::map<std::string, Eigen::MatrixXd>& gph2,
                    std::map<std::string, std::string>& params)
{
    return Ct;
}

