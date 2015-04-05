#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;


MatrixXd normalize_bistochastic(const MatrixXd& X, double tol, int max_iters)
{
    int n = X.rows();
    int m = X.cols();
    MatrixXd X1 = X;
    MatrixXd X2;
    VectorXd rowsums, colsums;

    for (int i = 0; i < max_iters; i++)
    {
        X2 = X1;

        // normalize rows
        rowsums = X1.rowwise().sum().array().inverse();
        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                X1(i,j) *= rowsums(i);

        // normalize columns
        colsums = X1.colwise().sum().array().inverse();
        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                X1(i,j) *= colsums(j);

        // stop condition
        double score = 0;
        double temp = 0;
        for (int i = 0; i < X.size(); i++) {
            temp = X1(i) - X2(i);
            score += temp*temp;
        }
        score = sqrt(score);
        if (score < tol)
            break;
    }
    return X1;
}

