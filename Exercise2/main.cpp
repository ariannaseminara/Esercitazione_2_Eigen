#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace Eigen;

void solve_system(const MatrixXd& A, const VectorXd& b, const VectorXd& solution) {
    // PALU Decomposition
    PartialPivLU<MatrixXd> lu(A);
    Vector2d x_palu = lu.solve(b);

    // QR Decomposition
    HouseholderQR<MatrixXd> qr(A);
    Vector2d x_qr = qr.solve(b);

    //Stampo i risultato della decomposizione mettendo 5 numeri significativi
    std::cout << "Solution using PALU Decomposition:\n" << std::fixed << std::setprecision(5) << std::scientific << x_palu << std::endl;
    std::cout << "Solution using QR Decomposition:\n" << std::fixed << std::setprecision(5) << std::scientific << x_qr << std::endl;

    //Calcolo errore relativo
    double relative_error_palu = (x_palu - solution).norm() / solution.norm();
    double relative_error_qr = (x_qr - solution).norm() / solution.norm();

    std::cout << "Relative error using PALU Decomposition: " << std::fixed << std::setprecision(5) << std::scientific << relative_error_palu << std::endl;
    std::cout << "Relative error using QR Decomposition: " << std::fixed << std::setprecision(5) << std::scientific << relative_error_qr << std::endl;
}

int main()
{
    // System 1
    Matrix2d A1; //Creo una matrice che chiamo A1
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    Vector2d solution1;
    solution1 << -1, -1;

    std::cout << "System 1:" << std::endl;
    solve_system(A1, b1, solution1);

    // System 2
    Matrix2d A2; //Creo una matrice che chiamo A2 e subito dopo inserisco i valori
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    Vector2d solution2;
    solution2 << -1, -1;

    std::cout << "System 2:" << std::endl;
    solve_system(A2, b2, solution2);

    // System 3
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    Vector2d solution3;
    solution3 << -1, -1;

    std::cout << "System 3:" << std::endl;
    solve_system(A3, b3, solution3);

    return 0;
}
