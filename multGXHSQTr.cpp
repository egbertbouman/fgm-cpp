#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::MatrixXi;


MatrixXi indDou2Int(MatrixXd& Ind0, int& len, int k, int& m, int& n)
{
    MatrixXi Ind;
    
    // empty
    if (len < 0)
    {
        m = k;
        n = k;
        len = k;
    
        Ind = MatrixXi(2*len, 1);
        for (int p = 0; p < len; ++p)
        {
            Ind(p * 2) = p;
            Ind(p * 2 + 1) = p;
        }
        // printf("len %d %d %d %d %d %d %d\n", len, Ind[0], Ind[1], Ind[2], Ind[3], Ind[38], Ind[39]);

        // non-empty
    }
    else
    {
        m = (int) Ind0(len * 2);
        n = (int) Ind0(len * 2 + 1);
    
        Ind = MatrixXi(2*len, 1);
        for (int p = 0; p < len; ++p)
        {
            Ind(p * 2) = (int) Ind0(p * 2) - 1;
            Ind(p * 2 + 1) = (int) Ind0(p * 2 + 1) - 1;
        }
    }

    return Ind;
}

/* 
 * function val = multGXHSQTr(indG, X, indH, IndS, Q)
 *
 *   X  -  m x n
 *   Y  -  mS x nS
 *   S  -  mS x nS
 *   G  -  mG x nG
 *   H  -  mH x nH
 *
 * History
 *   create  -  Feng Zhou, 03-20-2009
 *   modify  -  Feng Zhou, 09-03-2010
 */
double multGXHSQTr(const MatrixXd& indG, MatrixXd& X, MatrixXd& indH, MatrixXd& IndS0, MatrixXd& Q)
{
    // indG
    int mG = indG.rows();

    // X
    int m = X.rows();
    int n = X.cols();
    
    // indH
    int nH = indH.cols();

    // IndS
    int lenS = IndS0.cols() - 1;
    int mS, nS;
    MatrixXi IndS = indDou2Int(IndS0, lenS, n, mS, nS);

    // printf("mG %d nG %d mH %d nH %d m %d n %d\n", mG, nG, mH, nH, m, n);

    // check the dimension
    if (mS != mG || nS != nH)
        throw std::invalid_argument("Incorrect dimension!");

    // val
    double val = 0;

    int pS, iS, jS, i, j, idxX, idxY;
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
        val += X(idxX) * Q(idxY);
    }

    // release memory
    IndS.resize(0,0);

    return val;
}
