#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

const int MATCH_MIN = 0;
const int MATCH_MAX = 1;

/*
 * readMatrix
 * read a file into a matrix object
 * file must have matrix dimension as first line
 * matrix must be square
 */
void readMatrix(const char* filename, MatrixXd& m) {
  ifstream fin(filename);
  
  if (!fin) {
    cout << "Cannot open file." << endl;
    return;
  }
  
  // read in matrix dimensions and
  // create a matrix of that size
  int n_rows, n_cols;
  fin >> n_rows >> n_cols;
  MatrixXd temp(n_rows, n_cols);
  
  // read elements into the temporary matrix
  for (int i=0; i<n_rows; i++) {
    for (int j=0; j<n_cols; j++) {
      fin >> temp(i,j);
    }
  }
  fin.close();
  
  // if the dimensions are equal (square matrix), we're done
  // else, have to figure out larger dimension and pad with matrix max
  if (n_rows == n_cols) {
    m = temp;
  }
  else {
    float max_elem = temp.maxCoeff(); // find the max element
    float dim = max(n_rows, n_cols); // find the dimension for the new, square matrix
    m.resize(dim,dim);
    // fill the matrix with the elements from temp and pad with max element
    for (int i=0; i < dim; i++) {
      for (int j=0; j < dim; j++) {
        if (i >= n_rows || j >= n_cols) m(i,j) = max_elem;
        else m(i,j) = temp(i,j);
      }
    }
  }
}

/*
 * reduce
 * reduces matrix based on row and column minimums
 */
void reduce(MatrixXd& m) {
  // subtract row minimum from each row
  for (int i=0; i<m.rows(); i++) {
    float minElement = m.row(i).minCoeff();
    VectorXd rMinusMin(m.rows());
    rMinusMin.fill(-minElement);
    m.row(i) += rMinusMin;
  }
}

/*
 * hasMark
 * if there is a starred/primed zero in the given row/col, returns it's index
 * else, returns -1
 */
int hasMark(VectorXd& v) {  
  for (int i=0; i<v.size(); i++) {
    if (v(i)) {
      return i;
    }
  }
  return -1;
}

/*
 * swapStarsAndPrimes
 * Swap stars and primes based on step 5 of Hungarian algorithm
 * Z0 is uncovered primed zero we've found
 * Z1 is the stared zero in the column of Z0 (if any)
 * Z2 is the primed zero in the row of Z1 (will always be one)
 * ...continue series until we reach a primed zero with no starred zero in its column
 * Unstar each starred zero, star each primed zero, erase all primes and uncover every line in the matrix
 */
void swapStarsAndPrimes(int i, int j, MatrixXd& stars, MatrixXd& primes) {
  int primeRow = i;
  int primeCol = j;
  
  bool done = false;
  while (!done) {
    // find row index of row that has a 0* in the same col as the current 0'
    VectorXd col = stars.col(primeCol);
    int starInPrimeColRow = hasMark(col); 
    
    if (starInPrimeColRow < 0) {
      // star the prime we're looking at
      primes(primeRow, primeCol) = 0;
      stars(primeRow, primeCol) = 1;
      done = true;
    }
    else {
      // find which col has a 0' in the same row as z1
      VectorXd row = primes.row(starInPrimeColRow);
      int primeInStarRowCol = hasMark(row);
      
      // star first primed zero
      primes(primeRow, primeCol) = 0;
      stars(primeRow, primeCol) = 1;
      //primes(starInPrimeColRow, primeInStarRowCol) = 0;
      //stars(starInPrimeColRow, primeInStarRowCol) = 1;
      
      // unstar starred zero
      stars(starInPrimeColRow, primeCol) = 0;
      
      // set index of last prime, will check it's column for 0*s next
      primeRow = starInPrimeColRow;
      primeCol = primeInStarRowCol;
    }
  }
  // clear primes
  primes.fill(0);
}

/*
 * findMatching
 * implementation of the Hungarian matching algorithm
 * referenced from: http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
 */
void findMatching(MatrixXd& m, MatrixXd& result, int type) {
  MatrixXd n = m; // make a copy of m for reducing
  int dim = n.rows(); // dimension of matrix, used for checking if we've reduced
                      // the matrix enough yet
  
  MatrixXd stars(m.rows(), m.cols()); // matrix for storing our "starred" 0s (0*)
  stars.fill(0);
  MatrixXd primes(m.rows(), m.cols()); // matrix for storing our "primed" 0s (0')
  primes.fill(0);
  VectorXd rowCover(m.rows()); // keep track of which rows are "covered"
  rowCover.fill(0);
  VectorXd colCover(m.cols()); // keep track of which columns are "covered"
  colCover.fill(0);
  
  // to do maximization rather than minimization, we have to
  // transform the matrix by subtracting every value from the maximum
  if (type == MATCH_MAX) {
    float max = n.maxCoeff();
    MatrixXd maxMat(n.rows(), n.cols());
    maxMat.fill(max);
    n = maxMat - n;
  }
  
  // Step 1 
  // Reduce matrix
  reduce(n);
  
  // Step 2
  // Find a zero in the matrix. If there is no starred zero in 
  // its row or column, star Z. Repeat for each element in the matrix.
  for (int i=0; i<n.rows(); i++) {
    for (int j=0; j<n.cols(); j++) {
      if (n(i,j) == 0 && !rowCover(i) && !colCover(j)) {
        stars(i,j) = 1;
        rowCover(i) = 1;
        colCover(j) = 1;
      }
    }
  }
  // covers need to be cleared for following steps
  rowCover.fill(0);
  colCover.fill(0);
  
  while (true) {
    // Step 3
    // Cover all columns that have a starred zero
    // If the number of columns with starred zeroes equals the matrix
    // dimensions, we are done! Otherwise, move on to step 4.
    step3:
    for (int j=0; j<n.cols(); j++) {
      VectorXd col = stars.col(j);
      if (hasMark(col) >= 0) {
        colCover(j) = 1;
      }
    }
    if (colCover.sum() == dim) {
      result = stars;
      return;
    }
    
    // Step 4
    // Find a non-covered zero and prime it
    step4:
    for (int i=0; i<n.rows(); i++) {
      for (int j=0; j<n.cols(); j++) {
        if (n(i,j) == 0 && !rowCover(i) && !colCover(j)) {
          primes(i,j) = 1;
          // if no starred zero in the row...
          VectorXd row = stars.row(i);
          if (hasMark(row) < 0) {
            // Step 5
            // swap stars and primes            
            swapStarsAndPrimes(i, j, stars, primes);
    
            // clear lines
            rowCover.fill(0);
            colCover.fill(0);
            
            goto step3;
          }
          else {
            // cover row
            rowCover(i) = 1;
            
            // uncover column of the starred zero in the same row
            int col = hasMark(row);
            colCover(col) = 0;
          }
        }
      }
    }
    
    // Step 6
    // Should now be no more uncovered zeroes
    // Get the minimum uncovered element
    float min = 1000000;
    for (int i=0; i<n.rows(); i++) {
      for (int j=0; j<n.cols(); j++) {
        if (!rowCover(i) && !colCover(j) && n(i,j) < min) {
          min = n(i,j);
        }
      }
    }
    
    // Subtract minimum from uncovered elements, add it to elements covered twice
    for (int i=0; i<n.rows(); i++) {
      for (int j=0; j<n.cols(); j++) {
        if (!rowCover(i) && !colCover(j)) {
          n(i,j) -= min;
        }
        else if (rowCover(i) && colCover(j)) {
          n(i,j) += min;
        }
      }
    }
    
    goto step4;
  }
}

/*
 * main()
 */
int main(int argc, char *argv[])
{
  if (argc != 2) {
    cout << "Usage: ./a.out [matrixfile]" << endl;
    return 0;
  }
  
  // times
    clock_t start, end;
  float elapsedTime;

  // random seed
  srand((unsigned)time(0));

    // start time
  start = clock();
  
  // read in the matrix file
  MatrixXd m(1,1);
  readMatrix(argv[1], m);
  
  // create an empty matrix to put the result in
  MatrixXd result(m.rows(), m.cols());
  result.fill(0);
  
  // run the Hungarian (Munkres) algorithm to find the maximal matching
  findMatching(m, result, MATCH_MIN);
  
  MatrixXd mask = m.cwiseProduct(result);
  int sum = mask.sum();
  
  cout << "ORIGINAL MATRIX" << endl;
  cout << m << endl;
  cout << endl;
  cout << "ASSIGNMENT RESULT" << endl;
  cout << result << endl;
  cout << endl;
  cout << "FINAL SUM" << endl;
  cout << sum << endl;
  
  
  // stop time
  end = clock();
    elapsedTime = (float)(end - start) / CLOCKS_PER_SEC;
  cout << "Completed in: " << elapsedTime << " s" << endl;
    
  return 0;
}
