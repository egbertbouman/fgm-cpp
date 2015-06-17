#ifndef HUNGARIAN_H_INCLUDED
#define HUNGARIAN_H_INCLUDED

#include <Eigen/Dense>


void findMatching(Eigen::MatrixXd& m, Eigen::VectorXi& assignment);

void step2a(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, Eigen::MatrixXd& starMatrix,
            Eigen::MatrixXd& newStarMatrix, Eigen::MatrixXd& primeMatrix, Eigen::VectorXi& coveredColumns,
            Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void step2b(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, Eigen::MatrixXd& starMatrix, 
            Eigen::MatrixXd& newStarMatrix, Eigen::MatrixXd& primeMatrix, Eigen::VectorXi& coveredColumns, 
            Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void step3(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, Eigen::MatrixXd& starMatrix, 
           Eigen::MatrixXd& newStarMatrix, Eigen::MatrixXd& primeMatrix, Eigen::VectorXi& coveredColumns, 
           Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void step4(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, Eigen::MatrixXd& starMatrix, 
           Eigen::MatrixXd& newStarMatrix, Eigen::MatrixXd& primeMatrix, Eigen::VectorXi& coveredColumns, 
           Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);

void step5(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, Eigen::MatrixXd& starMatrix, 
           Eigen::MatrixXd& newStarMatrix, Eigen::MatrixXd& primeMatrix, Eigen::VectorXi& coveredColumns, 
           Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void buildassignmentvector(Eigen::VectorXi& assignment, Eigen::MatrixXd& starMatrix, int nOfRows, int nOfColumns);

#endif
