#include <vector>
#include "util.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


MatrixXd resize(const MatrixXd& m, int rows, int cols, double default_value)
{
    if (m.rows() > rows || m.cols() > cols)
        throw std::invalid_argument("Shrinking a matrix is currently not supported");

    auto result = MatrixXd(rows, cols);
    result.fill(default_value);
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
            result(i, j) = m(i, j);
    }
    return result;
}

MatrixXd mat2ind(const MatrixXd& m)
{
    int k = 0;
    MatrixXd result(2, m.size() + 1);
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
            if (m(i, j) != 0)
            {
                result(1, k) = i+1;
                result(0, k) = j+1;
                ++k;
            }
        }
    }
    result(1, k) = m.rows();
    result(0, k) = m.cols();
    result.conservativeResize(2, k + 1);
    return result;
}

MatrixXd mat2indC(const MatrixXd& m)
{
    MatrixXd result(1, m.cols());
    result.fill(0);

    for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
            if (m(i, j) != 0)
            {
                result(j) = i+1;
                break;
            }
    return result;
}

MatrixXd multDiag(const MatrixXd& A, const VectorXd& v)
{
    int m = A.rows();
    MatrixXd V = v.adjoint().replicate(m, 1);
    return A.cwiseProduct(V);
}

int sub2ind(int dimrow, int dimcol, int row, int col)
{
    return dimrow*col + row;
}

VectorXi sub2ind(int dimrow, int dimcol, const VectorXi setrow, const VectorXi setcol)
{
    VectorXi genidx(setrow.rows());
    for (int i = 0; i < setrow.rows(); i++)
    {
        genidx(i) = sub2ind(dimrow, dimcol, setrow(i), setcol(i));
    }
    return genidx;
}

template <typename Derived>
VectorXi find(const Eigen::DenseBase<Derived>& m)
{
    std::vector<int> res;
    for (int i=0; i < m.rows(); i++)
    {
        if (m(i) != 0)
        {
            res.push_back(i);
        }
    }
    Eigen::Map<VectorXi> result(res.data(), res.size());
    return result;
}

template VectorXi find<MatrixXd>(const Eigen::DenseBase<MatrixXd>& m);
template VectorXi find<VectorXi>(const Eigen::DenseBase<VectorXi>& m);

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
