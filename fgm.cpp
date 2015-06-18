#include <vector>
#include <iostream>
#include <Eigen/Sparse>

#include "util.hpp"
#include "hungarian.hpp"
#include "fgm.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseMatrix;


MatrixXd gmPosDHun(MatrixXd& X)
{
    double max_oeff = X.maxCoeff();
    X *= -1;
    X.array() += max_oeff;

    int n1 = X.rows();
    int n2 = X.cols();

    VectorXi result_vector(n1);
    result_vector.fill(0);
    findMatching(X, result_vector);

    // index -> matrix
    VectorXi idx;
    if (n1 <= n2)
    {
        idx = sub2ind(n1, n2, VectorXi::LinSpaced(n1, 0, n1-1), result_vector.adjoint());
    }
    else
    {
        VectorXi temp1 = find(result_vector);
        VectorXi temp2(temp1.size());
        for (int i = 0; i < temp1.size(); i++)
        {
            temp2(i) = result_vector(temp1(i));
        }
        idx = sub2ind(n1, n2, temp1.adjoint(), temp2.adjoint());
    }

    MatrixXd result(n1, n2);
    result.fill(0);
    for (int i = 0; i < idx.size(); i++)
    {
        result(idx(i)) = 1;
    }
    return result;
}

double multGXHSQTr(const MatrixXd& indG, const MatrixXd& X, const MatrixXd& indH, const MatrixXd& IndS0, const MatrixXd& Q)
{
    int n = X.cols();

    MatrixXi IndS;
    int mS, nS;
    int lenS = IndS0.cols() - 1;

    if (lenS < 0)
    {
        mS = nS = n;
        IndS = MatrixXi(2*n, 1);
        for (int p = 0; p < n; ++p)
        {
            IndS(p * 2) = p;
            IndS(p * 2 + 1) = p;
        }
    }
    else
    {
        mS = (int) IndS0(lenS * 2);
        nS = (int) IndS0(lenS * 2 + 1);
        IndS = MatrixXi(2*lenS, 1);
        for (int p = 0; p < lenS; ++p)
        {
            IndS(p * 2) = (int) IndS0(p * 2) - 1;
            IndS(p * 2 + 1) = (int) IndS0(p * 2 + 1) - 1;
        }
    }

    // check the dimension
    if (mS != indG.rows() || nS != indH.cols())
        throw std::invalid_argument("Incorrect dimension!");

    double result = 0;
    int iS, jS, i, j, idxX, idxY;
    for (int pS = 0; pS < lenS; ++pS)
    {
        iS = IndS((pS << 1));
        jS = IndS((pS << 1) + 1);

        i = (int) indG(iS);
        i--;
        j = (int) indH(jS);
        j--;

        if (i < 0 || j < 0)
            continue;

        idxY = jS * mS + iS;
        idxX = j * n + i;
        result += X(idxX) * Q(idxY);
    }

    return result;
}

std::pair<MatrixXd, double> fgm(MatrixXd& KP, MatrixXd& KQ, MatrixXd& Ct, MatrixXd& asgTX,
                                std::map<std::string, MatrixXd>& gph1,
                                std::map<std::string, MatrixXd>& gph2,
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
    MatrixXd X(Ct.rows(), Ct.cols());
    double iv = 1 + eps;
    for (int i = 0; i < X.rows(); ++i)
    {
        for (int j = 0; j < X.cols(); ++j)
            X(i, j) = Ct(i, j) == 0 ? 0 : iv;
    }

    double tol = 1e-7;
    n1 = X.rows();
    n2 = X.cols();

    if (n1 != n2)
    {
        // non-square
        int n_max = (n1 > n2) ? n1 : n2;
        MatrixXd Xslack = resize(X, n_max, n_max, 1);

        Xslack = normalize_bistochastic(Xslack, tol, 1000);
        Xslack.conservativeResize(n1, n2);
        X = Xslack;
    }
    else
    {
        // square
        X = normalize_bistochastic(X, tol, 1000);
    }

    SparseMatrix<double> Xs, GXGs, HXHs;
    SparseMatrix<double> GHHQQGs = GHHQQG.sparseView();
    SparseMatrix<double> KPs = KP.sparseView();
    double tmp1, tmp2;

    // path-following
    for (int iAlp = 0; iAlp < nAlp; ++iAlp)
    {
        // scale of alpha
        double alp = alps(iAlp);
    
        // MFW
        std::vector<SparseMatrix<double>> Ys(nHst);
        Xs = X.sparseView();

        // main iteration
        for (int nIt = 0; nIt < nItMa; ++nIt)
        {
            // gradient
            GXGs = G1s.adjoint() * Xs * G2s;
            HXHs = H1s.adjoint() * Xs * H2s;
            SparseMatrix<double> GrGm = KPs + H1s * GXGs.cwiseProduct(KQ) * H2s.adjoint() + G1s * HXHs.cwiseProduct(KQ) * G2s.adjoint();
            SparseMatrix<double> GrCon = 2 * GHHQQGs * Xs;
            SparseMatrix<double> Gr = GrGm + (alp - .5) * GrCon;

            // optimal direction
            MatrixXd Gr_temp = MatrixXd(Gr);
            SparseMatrix<double> Y = gmPosDHun(Gr_temp).sparseView();
            SparseMatrix<double> V = Y - Xs;

            // save to history
            int pHst = nIt % nHst;
            Ys[pHst] = Y / nHst;

            // alternative direction
            if (nIt - 1 >= nHst)
            {
                SparseMatrix<double> W = -Xs;
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
            double bGm = KPs.cwiseProduct(V).sum() + (GXGs.cwiseProduct(HYHs) + GYGs.cwiseProduct(HXHs)).cwiseProduct(KQ).sum();

            MatrixXd YY = V * MatrixXd(V.adjoint());
            MatrixXd XY = Xs * MatrixXd(V.adjoint());

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
            X = Xs + t * V;

            // stop condition
            if ((X.sparseView() - Xs).norm() < eps || t < eps)
                break;

            // store
            Xs = X.sparseView();
        }
    }

    // re-size to the original size
    X.conservativeResize(ns[0], ns[1]);

    //accuracy
    double acc = 0;
    if (asgTX.size() > 0)
    {
        int co = 0;
        VectorXi idx = find(asgTX);
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
