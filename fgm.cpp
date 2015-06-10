#include <map>  
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "normalize_bistochastic.cpp"
#include "multGXHSQTr.cpp"
#include "hungarian.cpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
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

MatrixXd multDiag(MatrixXd& A, VectorXd& v)
{
    int m = A.rows();
    MatrixXd V = v.adjoint().replicate(m, 1);
    return A.cwiseProduct(V);
}

int sub2ind(int dimrow, int dimcol, int row, int col)
{
    return dimrow*col + row;
}


VectorXd sub2ind(int dimrow, int dimcol, VectorXd setrow, VectorXd setcol)
{
    VectorXd genidx(setrow.rows());
    for (int i = 0; i < setrow.rows(); i++)
    {
        genidx(i) = sub2ind(dimrow, dimcol, setrow(i), setcol(i));
    }
    return genidx;
}

VectorXd find(const VectorXd a)
{
    std::vector<double> res;
    for (int i=0; i < a.rows(); i++)
    {
        if (a(i) != 0)
        {
            res.push_back(i);
        }
    }
    Eigen::Map<VectorXd> result(res.data(), res.size());
    return result;
}

MatrixXd gmPosDHun(MatrixXd& X0)
{
    double max_oeff = X0.maxCoeff();
    X0 *= -1;
    X0.array() += max_oeff;

    int n1 = X0.rows();
    int n2 = X0.cols();

    VectorXd result_vector(n1);
    result_vector.fill(0);
    findMatching(X0, result_vector);

    // index -> matrix
    VectorXd idx;
    if (n1 <= n2)
    {
        idx = sub2ind(n1, n2, VectorXd::LinSpaced(n1, 0, n1-1), result_vector.adjoint());
    }
    else
    {
        VectorXd temp1 = find(result_vector);
        VectorXd temp2(temp1.size());
        for (int i = 0; i < temp1.size(); i++)
        {
            temp2(i) = result_vector(temp1(i));
        }
        idx = sub2ind(n1, n2, temp1.adjoint(), temp2.adjoint());
    }

    MatrixXd X(n1, n2);
    X.fill(0);
    for (int i = 0; i < idx.size(); i++)
    {
        X(idx(i)) = 1;
    }
    return X;
}

double pathDObjGm(MatrixXd& X, SparseMatrix<double>& G1s, SparseMatrix<double>& G2s, SparseMatrix<double>& H1s, SparseMatrix<double>& H2s, MatrixXd& KP, MatrixXd& KQ)
{
    SparseMatrix<double> Xs = X.sparseView();
    SparseMatrix<double> GXGs = G1s.adjoint() * Xs * G2s;
    SparseMatrix<double> HXHs = H1s.adjoint() * Xs * H2s;

    double tmp1 = KP.sparseView().cwiseProduct(Xs).sum();
    double tmp2 = GXGs.cwiseProduct(HXHs).cwiseProduct(KQ).sum();
    return tmp1 + tmp2;
}

std::pair<MatrixXd, double> fgm(MatrixXd& KP, MatrixXd& KQ, MatrixXd& Ct, MatrixXd& asgTX,
             std::map<std::string, MatrixXd>& gph1, std::map<std::string, MatrixXd>& gph2,
             std::map<std::string, std::string>& params)
{
    // TODO : get params from python
    int nAlp = 101;
    int nItMa = 100;
    int nHst = 10;

    // weight
    VectorXd alps = VectorXd::LinSpaced(nAlp, 0, 1);

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

    // add additional nodes to make sure n1 == n2
    if (n1 < n2)
    {
        auto mi = KP.minCoeff();
        KP = resize(KP, n2, n2, mi);
        G1 = resize(G1, n2, m1, 0);
        H1 = resize(H1, n2, m1, 0);
        Ct = resize(Ct, n2, n2, 1);
    }
    else if (n1 > n2)
    {
        auto mi = KP.minCoeff();
        KP = resize(KP, n1, n1, mi);
        G2 = resize(G2, n1, m2, 0);
        H2 = resize(H2, n1, m2, 0);
        Ct = resize(Ct, n1, n1, 1);
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

    // factorize KQ using SVD
    Eigen::JacobiSVD<MatrixXd> svd(KQ, Eigen::ComputeFullU | Eigen::ComputeFullV);

    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    VectorXd s = svd.singularValues();
    int length = s.size();

    MatrixXd Us = U.leftCols(length);
    MatrixXd Vs = V.leftCols(length);
    VectorXd s_sqrt = s.head(length).cwiseSqrt().real();
    MatrixXd XQ1 = multDiag(Us, s_sqrt).adjoint();
    MatrixXd XQ2 = multDiag(Vs, s_sqrt).adjoint();

    // auxiliary variables for computing the derivative of the constant term
    MatrixXd QQ1 = XQ1.adjoint() * XQ1;
    MatrixXd QQ2 = XQ2.adjoint() * XQ2;
    MatrixXd GHHQQG = G1 * HH1.cwiseProduct(QQ1) * G1.adjoint() + G2 * HH2.cwiseProduct(QQ2) * G2.adjoint();

    // initialize from a doubly stochastic matrix
    double eps = std::numeric_limits<double>::epsilon();
    MatrixXd X0 = MatrixXd(Ct.rows(), Ct.cols());
    double iv = 1 + eps;
    for (int i = 0; i < X0.rows(); ++i)
    {
        for (int j = 0; j < X0.cols(); ++j)
            X0(i, j) = Ct(i, j) == 0 ? 0 : iv;
    }

    double tol = 1e-7;
    n1 = X0.rows();
    n2 = X0.cols();
    MatrixXd X;

    if (n1 != n2)
    {
        // non-square
        int n_max = (n1 > n2) ? n1 : n2;
        MatrixXd Xslack = resize(X0, n_max, n_max, 1);

        Xslack = normalize_bistochastic(Xslack, tol, 1000);
        Xslack.conservativeResize(n1, n2);
        X = Xslack;
    }
    else
    {
        // square
        X0 = normalize_bistochastic(X0, tol, 1000);
    }

    SparseMatrix<double> Xs = X0.sparseView();
    SparseMatrix<double> GXGs, HXHs;
    double tmp1, tmp2;


    // path-following
    for (int iAlp = 0; iAlp < nAlp; ++iAlp)
    {
        // scale of alpha
        double alp = alps(iAlp);
    
        // MFW
        vector<SparseMatrix<double>> Ys(nHst);
        SparseMatrix<double> X0s = X0.sparseView();

        // main iteration
        for (int nIt = 1; nIt < nItMa; ++nIt)
        {
            // gradient
            GXGs = G1s.adjoint() * X0s * G2s;
            HXHs = H1s.adjoint() * X0s * H2s;
            SparseMatrix<double> GrGm = KP.sparseView() + H1s * GXGs.cwiseProduct(KQ) * H2s.adjoint() + G1s * HXHs.cwiseProduct(KQ) * G2s.adjoint();
            SparseMatrix<double> GHHQQGs = GHHQQG.sparseView();
            SparseMatrix<double> GrCon = 2 * GHHQQGs * X0s;
            SparseMatrix<double> Gr = GrGm + (alp - .5) * GrCon;

            // optimal direction
            SparseMatrix<double> Y = gmPosDHun(MatrixXd(Gr)).sparseView();
            SparseMatrix<double> V = Y - X0s;

            // save to history
            int pHst = (nIt - 1) % nHst;
            Ys[pHst] = Y / nHst;

            // alternative direction
            if (nIt >= nHst)
            {
                SparseMatrix<double> W = -X0s;
                for (int iHst = 0; iHst < nHst; ++iHst)
                    W = W + Ys[iHst];

                double vV = Gr.cwiseProduct(V).sum() / V.norm();
                double vW = Gr.cwiseProduct(W).sum() / W.norm();
                if (vW > vV)
                {
                    V = W;
                    Ys[pHst] = Y / nHst;
                }
            }

            // step size
            SparseMatrix<double> GYGs = G1s.adjoint() * V * G2s;
            SparseMatrix<double> HYHs = H1s.adjoint() * V * H2s;
            double aGm = GYGs.cwiseProduct(HYHs).cwiseProduct(KQ).sum();
            double bGm = KP.sparseView().cwiseProduct(V).sum() + (GXGs.cwiseProduct(HYHs) + GYGs.cwiseProduct(HXHs)).cwiseProduct(KQ).sum();

            MatrixXd YY = V * MatrixXd(V.adjoint());
            MatrixXd XY = X0s * MatrixXd(V.adjoint());

            tmp1 = multGXHSQTr(indG1.adjoint(), YY, indG1, IndHH1, QQ1);
            tmp2 = multGXHSQTr(indG2.adjoint(), YY, indG2, IndHH2, QQ2);
            double aCon = tmp1 + tmp2;

            tmp1 = multGXHSQTr(indG1.adjoint(), XY, indG1, IndHH1, QQ1);
            tmp2 = multGXHSQTr(indG2.adjoint(), XY, indG2, IndHH2, QQ2);
            double bCon = 2 * (tmp1 + tmp2);

            double a = aGm + (alp - .5) * aCon;
            double b = bGm + (alp - .5) * bCon;

            double t = -b / a / 2;

            if (t <= 0)
            {
                t = (a > 0) ? 1 : 0;
            }
            else if (t <= 0.5)
            {
                if (a > 0)
                    t = 1;
            }
            else if (t <= 1)
            {
                if (a > 0)
                    t = 1;
            }
            else
            {
                t = (a > 0) ? 0 : 1;
            }
            
            // update
            X = X0s + t * V;

            // stop condition
            if ((X.sparseView() - X0s).norm() < eps || t < eps)
                break;

            // store
            X0s = X.sparseView();
        }

        // store
        X0 = X;

    }

    // re-size to the original size
    X.conservativeResize(ns[0], ns[1]);

    //accuracy
    double acc = 0;
    if (asgTX.size() > 0)
    {
        int co = 0;
        VectorXd idx = find(asgTX);
        for (int i = 0; i < idx.size(); ++i)
        {
            // correct correspondences found
            if (asgTX(idx(i)) == X(idx(i)))
                co += 1;
        }
        // percentage
        acc = co / (double)idx.size();
    }

    std::pair<MatrixXd, double> result(X, acc);
    return result;
}

