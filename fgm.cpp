#include <map>  
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using Eigen::MatrixXd;
using Eigen::SparseMatrix;


MatrixXd resize(const MatrixXd& m, int rows, int cols, double default)
{
    if (m.rows() > rows || m.cols() > cols)
        throw std::invalid_argument("Shrinking a matrix is currently not supported");

    auto result = MatrixXd(rows, cols);
    result.fill(default);
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
                result(0, k) = i;
                result(1, k) = j;
                ++k;
            }
        }
    }
    result(0, k) = m.rows();
    result(1, k) = m.cols();
    result.conservativeResize(2, k + 1);
    return result;
}

MatrixXd mat2indC(const MatrixXd& m)
{
    std::vector<double> cols;
    for (int j = 0; j < m.cols(); ++j)
    {
        for (int i = 0; i < m.rows(); ++i)
        {
            if (m(i, j) != 0)
            {
                cols.push_back(j);
                break;
            }
        }
    }
    Eigen::Map<MatrixXd> result(cols.data(), cols.size(), 1);
    return result;
}

MatrixXd fgm(MatrixXd& KP, MatrixXd& KQ, MatrixXd& Ct, MatrixXd& asgTX,
             std::map<std::string, MatrixXd>& gph1, std::map<std::string, MatrixXd>& gph2,
             std::map<std::string, std::string>& params)
{
    // TODO : get params from python
    int nAlp = 101;
    int nItMa = 100;
    int nHst = 10;
    bool isIp = false;
    bool isDeb = false;

    // weight
    Eigen::VectorXd alps = Eigen::VectorXd::LinSpaced(nAlp, 0, 1);

    // graph elements
    MatrixXd G1 = gph1["G"];
    MatrixXd H1 = gph1["H"];
    MatrixXd G2 = gph2["G"];
    MatrixXd H2 = gph2["H"];

    // dimension
    int n1 = G1.rows();
    int m1 = G1.cols();
    int n2 = G2.rows();
    int m2 = G2.cols();
    int ns[] = { n1, n2 };

    MatrixXd zero = MatrixXd::Zero(2, 3);
    zero << 1, 0, 0, 0, 1, 0;
    std::cout << zero << std::endl << std::endl << mat2ind(zero) << std::endl << std::endl << mat2indC(zero);

    // add additional nodes to make sure n1 == n2
    if (n1 < n2)
    {
        auto mi = KP.minCoeff();
        KP = resize(KP, n2, n2, mi);
        G1 = resize(G1, n2, m1, 0);
        H1 = resize(H1, n2, m1, 0);
        Ct = resize(Ct, n2, n2, 1);
        //if !isempty(XT)
        //	XT = [XT; zeros(n2 - n1, n2)];
    }
    else if (n1 > n2)
    {
        auto mi = KP.minCoeff();
        KP = resize(KP, n1, n1, mi);
        G2 = resize(G2, n1, m2, 0);
        H2 = resize(H2, n1, m2, 0);
        Ct = resize(Ct, n1, n1, 1);
        //if !isempty(XT)
        //	XT = [XT, zeros(n1, n1 - n2)];
    }

    // auxiliary variables (for saving computational time)
    MatrixXd GG1 = G1.adjoint() * G1;
    MatrixXd GG2 = G2.adjoint() * G2;
    MatrixXd HH1 = H1.adjoint() * H1;
    MatrixXd HH2 = H2.adjoint() * H2;
    MatrixXd IndHH1 = mat2ind(HH1);
    MatrixXd IndHH2 = mat2ind(HH2);

    MatrixXd indG1 = mat2indC(G1);
    MatrixXd indG2 = mat2indC(G2);
    MatrixXd indH1 = mat2indC(H1);
    MatrixXd indH2 = mat2indC(H2);

    // sparse matrix
    SparseMatrix<double> G1s = G1.sparseView();
    SparseMatrix<double> G2s = G2.sparseView();
    SparseMatrix<double> H1s = H1.sparseView();
    SparseMatrix<double> H2s = H2.sparseView();
    MatrixXd HH1s = H1s.adjoint() * H1s;
    MatrixXd HH2s = H2s.adjoint() * H2s;

    //std::cout << alps << std::endl;

    return Ct;
}

