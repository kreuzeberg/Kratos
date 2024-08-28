//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#pragma once

// System includes

// External includes

// Project includes
#include "trilinos_space_experimental.h"

namespace Kratos
{
///@addtogroup KratosCore
///@{

///@name Kratos Classes
///@{

/**
 * @class TrilinosCPPTestExperimentalUtilities
 * @brief Utilities to develop C++ tests in Trilinos
 * @ingroup TrilinosApplication
 * @author Vicente Mataix Ferandiz
 */
class TrilinosCPPTestExperimentalUtilities
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of TrilinosDofUpdater
    KRATOS_CLASS_POINTER_DEFINITION(TrilinosCPPTestExperimentalUtilities);

    /// Basic definitions
    using TrilinosSparseMatrix = Tpetra::FECrsMatrix<>;
    using TrilinosSparseVector = Tpetra::Vector<>;
    using TrilinosSparseSpaceType = TrilinosSpaceExperimental<TrilinosSparseMatrix, TrilinosSparseVector>;
    using TrilinosLocalSpaceType = UblasSpace<double, Matrix, Vector>;

    /// Trilinos definitions
    using TrilinosSparseMatrixType = TrilinosSparseSpaceType::MatrixType;
    using TrilinosVectorType = TrilinosSparseSpaceType::VectorType;

    using TrilinosSparseMatrixSmartReferenceType = Teuchos::RCP<TrilinosSparseMatrix>;
    using TrilinosVectorSmartReferenceType = Teuchos::RCP<TrilinosVectorType>;

    using TrilinosLocalMatrixType = TrilinosLocalSpaceType::MatrixType;
    using TrilinosLocalVectorType = TrilinosLocalSpaceType::VectorType;

    /// Define the graph type
    using GraphType = Tpetra::FECrsGraph<>;

    /// Define the map type
    using MapType = Tpetra::Map<>;

    /// Tpetra definitions
    // Your scalar type; the type of sparse matrix entries. e.g., double.
    using ST = Tpetra::FECrsMatrix<>::scalar_type;
    // Your local ordinal type; the signed integer type 
    // used to store local sparse matrix indices.  e.g., int.
    using LO = MapType::local_ordinal_type;
    // Your global ordinal type; the signed integer type 
    // used to index the matrix globally, over all processes.
    // e.g., int, long, ptrdif_t, int64_t, ...
    using GO = MapType::global_ordinal_type;
    // The Node type.  e.g., Kokkos::DefaultNode::DefaultNodeType, 
    // defined in KokkosCompat_DefaultNode.hpp.
    using NT = MapType::node_type;

    ///@}
    ///@name Life Cycle
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
    * @brief Generates a dummy diagonal local matrix for Trilinos
    * @param rDataCommunicator The data communicator considered
    * @param NumGlobalElements The global dimension of the matrix
    * @param Offset The offset considered
    * @param AddNoDiagonalValues If adding non diagonal values
    */
    static TrilinosLocalMatrixType GenerateDummyLocalMatrix(
        const int NumGlobalElements = 12,
        const double Offset = 0.0,
        const bool AddNoDiagonalValues = false
        );

    /**
    * @brief Generates a dummy diagonal sparse matrix for Trilinos
    * @param rDataCommunicator The data communicator considered
    * @param NumGlobalElements The global dimension of the matrix
    * @param Offset The offset considered
    * @param AddNoDiagonalValues If adding non diagonal values
    */
    static TrilinosSparseMatrixSmartReferenceType GenerateDummySparseMatrix(
        const DataCommunicator& rDataCommunicator,
        const int NumGlobalElements = 12,
        const double Offset = 0.0,
        const bool AddNoDiagonalValues = false
        );

    /**
    * @brief Generates a dummy local vector for Trilinos
    * @param rDataCommunicator The data communicator considered
    * @param NumGlobalElements The global dimension of the matrix
    * @param Offset The offset considered
    */
    static TrilinosLocalVectorType GenerateDummyLocalVector(
        const int NumGlobalElements = 12,
        const double Offset = 0.0
        );

    /**
    * @brief Generates a dummy sparse vector for Trilinos
    * @param rDataCommunicator The data communicator considered
    * @param NumGlobalElements The global dimension of the matrix
    * @param Offset The offset considered
    */
    static TrilinosVectorSmartReferenceType GenerateDummySparseVector(
        const DataCommunicator& rDataCommunicator,
        const int NumGlobalElements = 12,
        const double Offset = 0.0
        );

    /**
    * @brief This method checks the values of a sparse vector with the given serial vector
    * @param vector The matrix to check
    * @param vector The reference matrix
    * @param NegligibleValueThreshold The tolerance considered
    */
    static void CheckSparseVectorFromLocalVector(
        const TrilinosVectorType& rA,
        const TrilinosLocalVectorType& rB,
        const double NegligibleValueThreshold = 1e-8
        );

    /**
    * @brief This method checks the values of a sparse vector with the given indices and values
    * @param rb The vector to check
    * @param rIndexes The indices
    * @param rValues The values
    * @param NegligibleValueThreshold The tolerance considered
    */
    static void CheckSparseVector(
        const TrilinosVectorType& rb,
        const std::vector<int>& rIndexes,
        const std::vector<double>& rValues,
        const double NegligibleValueThreshold = 1e-8
        );

    /**
    * @brief This method checks the values of a sparse matrix with the given serial matrix
    * @param rA The matrix to check
    * @param rB The reference matrix
    * @param NegligibleValueThreshold The tolerance considered
    */
    static void CheckSparseMatrixFromLocalMatrix(
        const TrilinosSparseMatrixType& rA,
        const TrilinosLocalMatrixType& rB,
        const double NegligibleValueThreshold = 1e-8
        );

    /**
    * @brief This method checks the values of a sparse matrix with the given indices and values
    * @param rA The matrix to check
    * @param rRowIndexes The row indices
    * @param rColumnIndexes The column indices
    * @param rB The reference matrix
    * @param NegligibleValueThreshold The tolerance considered
    */
    static void CheckSparseMatrixFromLocalMatrix(
        const TrilinosSparseMatrixType& rA,
        const std::vector<int>& rRowIndexes,
        const std::vector<int>& rColumnIndexes,
        const TrilinosLocalMatrixType& rB,
        const double NegligibleValueThreshold = 1e-8
        );

    /**
    * @brief This method checks the values of a sparse matrix with the given indices and values
    * @param rA The matrix to check
    * @param rRowIndexes The row indices
    * @param rColumnIndexes The column indices
    * @param rValues The values
    * @param NegligibleValueThreshold The tolerance considered
    */
    static void CheckSparseMatrix(
        const TrilinosSparseMatrixType& rA,
        const std::vector<int>& rRowIndexes,
        const std::vector<int>& rColumnIndexes,
        const std::vector<double>& rValues,
        const double NegligibleValueThreshold = 1e-8
        );

    /**
    * @brief This method generates a set of row, columns and values from a sparse matrix
    * @param rA The matrix where generate the vectors of rows, indexes and values
    * @param rRowIndexes The row indices
    * @param rColumnIndexes The column indices
    * @param rValues The values
    * @param PrintValues If printing the vectors, for debugging purposes
    * @param IncludeHardZeros If including the hard zeros (giving  a certain threshold)
    */
    static void GenerateSparseMatrixIndexAndValuesVectors(
        const TrilinosSparseSpaceType::MatrixType& rA,
        std::vector<int>& rRowIndexes,
        std::vector<int>& rColumnIndexes,
        std::vector<double>& rValues,
        const bool PrintValues = false,
        const double ThresholdIncludeHardZeros = -1
        );

    /**
    * @brief This method generates a sparse matrix from a set of row, columns and values
    * @param rDataCommunicator The data communicator considered
    * @param NumGlobalElements The global dimension of the matrix
    * @param rRowIndexes The row indices
    * @param rColumnIndexes The column indices
    * @param rValues The values
    * @param pMap Map pointer
    * @return The matrix generated
    */
    static TrilinosSparseMatrixSmartReferenceType GenerateSparseMatrix(
        const DataCommunicator& rDataCommunicator,
        const int NumGlobalElements,
        const std::vector<int>& rRowIndexes,
        const std::vector<int>& rColumnIndexes,
        const std::vector<double>& rValues,
        const MapType* pMap =  nullptr
        );

    ///@}

}; /// class TrilinosCPPTestExperimentalUtilities

} /// namespace Kratos