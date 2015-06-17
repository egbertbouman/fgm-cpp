#ifndef FGM_H_INCLUDED
#define FGM_H_INCLUDED

#include <utility>
#include <string>
#include <map>
#include <Eigen/Dense>

#define FGM_VERSION "0.0.1"


Eigen::MatrixXd gmPosDHun(Eigen::MatrixXd& X);

double multGXHSQTr(const Eigen::MatrixXd& indG, const Eigen::MatrixXd& X,
                   const Eigen::MatrixXd& indH, const Eigen::MatrixXd& IndS0, const Eigen::MatrixXd& Q);

std::pair<Eigen::MatrixXd, double> fgm(Eigen::MatrixXd& KP, Eigen::MatrixXd& KQ, 
                                       Eigen::MatrixXd& Ct, Eigen::MatrixXd& asgTX, 
                                       std::map<std::string, Eigen::MatrixXd>& gph1,
                                       std::map<std::string, Eigen::MatrixXd>& gph2,
                                       std::map<std::string, std::string>& params);

#endif
