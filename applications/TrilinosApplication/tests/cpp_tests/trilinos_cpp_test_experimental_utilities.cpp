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

// System includes

// External includes

// Project includes
#include "includes/expect.h"
#include "trilinos_cpp_test_utilities.h"
#include "trilinos_cpp_test_experimental_utilities.h"

namespace Kratos
{

TrilinosCPPTestExperimentalUtilities::TrilinosLocalMatrixType TrilinosCPPTestExperimentalUtilities::GenerateDummyLocalMatrix(
    const int NumGlobalElements,
    const double Offset,
    const bool AddNoDiagonalValues
    )
{
    return TrilinosCPPTestUtilities::GenerateDummyLocalMatrix(NumGlobalElements, Offset, AddNoDiagonalValues);
}

/***********************************************************************************/
/***********************************************************************************/

TrilinosCPPTestExperimentalUtilities::TrilinosSparseMatrixSmartReferenceType TrilinosCPPTestExperimentalUtilities::GenerateDummySparseMatrix(
    const DataCommunicator& rDataCommunicator,
    const int NumGlobalElements,
    const double Offset,
    const bool AddNoDiagonalValues
    )
{
    // Define the graph type
    using graph_type = Tpetra::FECrsGraph<>;  // Define the graph type

    // Generate Tpetra communicator
    KRATOS_ERROR_IF_NOT(rDataCommunicator.IsDistributed()) << "Only distributed DataCommunicators can be used!" << std::endl;
    auto raw_mpi_comm = MPIDataCommunicator::GetMPICommunicator(rDataCommunicator);
    Teuchos::RCP<const Teuchos::Comm<int>> tpetra_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(raw_mpi_comm));

    // Create a map
    Tpetra::global_size_t numGlobalElements = static_cast<Tpetra::global_size_t>(NumGlobalElements);
    Teuchos::RCP<const Tpetra::Map<>> map = Teuchos::rcp(new Tpetra::Map<>(numGlobalElements, 0, tpetra_comm));
    KRATOS_ERROR_IF_NOT(map->getGlobalNumElements() == static_cast<std::size_t>(NumGlobalElements)) << "Inconsistent number of rows" << std::endl;

    // Create a graph
    const size_t maxNodeEntriesPerRow = AddNoDiagonalValues ? 3 : 1;
    Teuchos::RCP<graph_type> graph = Teuchos::rcp(new graph_type(map, map, maxNodeEntriesPerRow));

    // Begin graph assembly
    graph->resumeFill();

    const auto numMyElements = map->getNodeNumElements();
    auto myGlobalElements = map->getNodeElementList();
    std::vector<GO> nonDiagonalIndices(2); // Define based on your Tpetra types

    for (std::size_t i = 0; i < numMyElements; ++i) {
        // Non-diagonal values
        if (AddNoDiagonalValues) {
            if (myGlobalElements[i] == 0) {
                nonDiagonalIndices[0] = 1;
                graph->insertGlobalIndices(myGlobalElements[i], Teuchos::ArrayView<const GO>(&nonDiagonalIndices[0], 1));
            } else if (myGlobalElements[i] == NumGlobalElements - 1) {
                nonDiagonalIndices[0] = NumGlobalElements - 2;
                graph->insertGlobalIndices(myGlobalElements[i], Teuchos::ArrayView<const GO>(&nonDiagonalIndices[0], 1));
            } else {
                nonDiagonalIndices[0] = myGlobalElements[i] - 1;
                nonDiagonalIndices[1] = myGlobalElements[i] + 1;
                graph->insertGlobalIndices(myGlobalElements[i], Teuchos::ArrayView<const GO>(nonDiagonalIndices.data(), 2));
            }
        }

        // Insert diagonal entry
        graph->insertGlobalIndices(myGlobalElements[i], Teuchos::ArrayView<const GO>(&myGlobalElements[i], 1));
    }

    // End graph assembly
    graph->fillComplete();

    // Create a Tpetra::FECrsMatrix using the graph
    Teuchos::RCP<Tpetra::FECrsMatrix<>> A = Teuchos::rcp(new Tpetra::FECrsMatrix<>(graph));

    // Define non-diagonal values and other necessary variables
    std::vector<double> nonDiagonalValues(2, -1.0);
    double value;

    // Begin matrix assembly
    A->beginAssembly();

    for (std::size_t i = 0; i < numMyElements; ++i) {
        // Non-diagonal values
        if (AddNoDiagonalValues) {
            if (myGlobalElements[i] == 0) {
                A->replaceGlobalValues(myGlobalElements[i], Teuchos::ArrayView<const GO>(&nonDiagonalIndices[0], 1), Teuchos::ArrayView<const double>(&nonDiagonalValues[0], 1));
            } else if (myGlobalElements[i] == NumGlobalElements - 1) {
                A->replaceGlobalValues(myGlobalElements[i], Teuchos::ArrayView<const GO>(&nonDiagonalIndices[0], 1), Teuchos::ArrayView<const double>(&nonDiagonalValues[0], 1));
            } else {
                A->replaceGlobalValues(myGlobalElements[i], Teuchos::ArrayView<const GO>(nonDiagonalIndices.data(), 2), Teuchos::ArrayView<const double>(nonDiagonalValues.data(), 2));
            }
        }

        // Insert diagonal entry
        value = Offset + static_cast<double>(myGlobalElements[i]);
        A->replaceGlobalValues(myGlobalElements[i], Teuchos::ArrayView<const GO>(&myGlobalElements[i], 1), Teuchos::ArrayView<const double>(&value, 1));
    }

    // End matrix assembly
    A->endAssembly();

    // Complete the fill process, optimizing data for matrix-vector products
    A->fillComplete();

    return A;
}

/***********************************************************************************/
/***********************************************************************************/

TrilinosCPPTestExperimentalUtilities::TrilinosLocalVectorType TrilinosCPPTestExperimentalUtilities::GenerateDummyLocalVector(
    const int NumGlobalElements,
    const double Offset
    )
{
    return TrilinosCPPTestUtilities::GenerateDummyLocalVector(NumGlobalElements, Offset);
}

// /***********************************************************************************/
// /***********************************************************************************/

// TrilinosCPPTestExperimentalUtilities::TrilinosVectorType TrilinosCPPTestExperimentalUtilities::GenerateDummySparseVector(
//     const DataCommunicator& rDataCommunicator,
//     const int NumGlobalElements,
//     const double Offset
//     )
// {
//     // Generate Epetra communicator
//     KRATOS_ERROR_IF_NOT(rDataCommunicator.IsDistributed()) << "Only distributed DataCommunicators can be used!" << std::endl;
//     auto raw_mpi_comm = MPIDataCommunicator::GetMPICommunicator(rDataCommunicator);
//     Epetra_MpiComm epetra_comm(raw_mpi_comm);

//     // Create a map
//     const Epetra_Map map(NumGlobalElements,0,epetra_comm);

//     // Local number of rows
//     const int NumMyElements = map.NumMyElements();

//     // Get update list
//     int* MyGlobalElements = map.MyGlobalElements( );

//     // Create an Epetra_Vector
//     TrilinosVectorType b(map);

//     double value;
//     for( int i=0 ; i<NumMyElements; ++i ) {
//         value = Offset + static_cast<double>(MyGlobalElements[i]);
//         b[0][i]= value;
//     }

//     b.GlobalAssemble();

//     return b;
// }

// /***********************************************************************************/
// /***********************************************************************************/

// void TrilinosCPPTestExperimentalUtilities::CheckSparseVectorFromLocalVector(
//     const TrilinosVectorType& rA,
//     const TrilinosLocalVectorType& rB,
//     const double NegligibleValueThreshold
//     )
// {
//     const std::size_t total_size = rB.size();
//     std::vector<int> indexes;
//     indexes.reserve(total_size);
//     std::vector<double> values;
//     values.reserve(total_size);
//     for (std::size_t i = 0; i < rB.size(); ++i) {
//         indexes.push_back(i);
//         values.push_back(rB[i]);
//     }
//     CheckSparseVector(rA, indexes, values);
// }

// /***********************************************************************************/
// /***********************************************************************************/

// void TrilinosCPPTestExperimentalUtilities::CheckSparseVector(
//     const TrilinosVectorType& rb,
//     const std::vector<int>& rIndexes,
//     const std::vector<double>& rValues,
//     const double NegligibleValueThreshold
//     )
// {
//     int index;
//     double value;
//     const auto& r_map = rb.Map();
//     for (std::size_t counter = 0; counter < rIndexes.size(); ++counter) {
//         index = rIndexes[counter];
//         value = rValues[counter];
//         if (r_map.MyGID(index)) {
//             const double ref_value = rb[0][r_map.LID(index)];
//             KRATOS_EXPECT_RELATIVE_NEAR(value, ref_value, NegligibleValueThreshold);
//         }
//     }
// }

// /***********************************************************************************/
// /***********************************************************************************/

// void TrilinosCPPTestExperimentalUtilities::CheckSparseMatrixFromLocalMatrix(
//     const TrilinosSparseMatrixType& rA,
//     const TrilinosLocalMatrixType& rB,
//     const double NegligibleValueThreshold
//     )
// {
//     const std::size_t total_size = rB.size1() * rB.size2();
//     std::vector<int> row_indexes;
//     row_indexes.reserve(total_size);
//     std::vector<int> column_indexes;
//     column_indexes.reserve(total_size);
//     std::vector<double> values;
//     values.reserve(total_size);
//     for (std::size_t i = 0; i < rB.size1(); ++i) {
//         for (std::size_t j = 0; j < rB.size2(); ++j) {
//             const double value = rB(i, j);
//             if (std::abs(value) > std::numeric_limits<double>::epsilon()) {
//                 row_indexes.push_back(i);
//                 column_indexes.push_back(j);
//                 values.push_back(value);
//             }
//         }
//     }
//     CheckSparseMatrix(rA, row_indexes, column_indexes, values);
// }

// /***********************************************************************************/
// /***********************************************************************************/

// void TrilinosCPPTestExperimentalUtilities::CheckSparseMatrixFromLocalMatrix(
//     const TrilinosSparseMatrixType& rA,
//     const std::vector<int>& rRowIndexes,
//     const std::vector<int>& rColumnIndexes,
//     const TrilinosLocalMatrixType& rB,
//     const double NegligibleValueThreshold
//     )
// {

//     int row, column;
//     std::vector<double> values(rRowIndexes.size());
//     for (std::size_t counter = 0; counter < rRowIndexes.size(); ++counter) {
//         row = rRowIndexes[counter];
//         column = rColumnIndexes[counter];
//         values[counter] = rB(row, column);
//     }
//     CheckSparseMatrix(rA, rRowIndexes, rColumnIndexes, values, NegligibleValueThreshold);
// }


// /***********************************************************************************/
// /***********************************************************************************/

// void TrilinosCPPTestExperimentalUtilities::CheckSparseMatrix(
//     const TrilinosSparseMatrixType& rA,
//     const std::vector<int>& rRowIndexes,
//     const std::vector<int>& rColumnIndexes,
//     const std::vector<double>& rValues,
//     const double NegligibleValueThreshold
//     )
// {
//     int local_validated_values = 0;
//     int row, column;
//     double value;
//     for (std::size_t counter = 0; counter < rRowIndexes.size(); ++counter) {
//         row = rRowIndexes[counter];
//         column = rColumnIndexes[counter];
//         value = rValues[counter];
//         for (int i = 0; i < rA.NumMyRows(); i++) {
//             int numEntries; // Number of non-zero entries
//             double* vals;   // Row non-zero values
//             int* cols;      // Column indices of row non-zero values
//             rA.ExtractMyRowView(i, numEntries, vals, cols);
//             const int row_gid = rA.RowMap().GID(i);
//             if (row == row_gid) {
//                 int j;
//                 for (j = 0; j < numEntries; j++) {
//                     const int col_gid = rA.ColMap().GID(cols[j]);
//                     if (col_gid == column) {
//                         KRATOS_EXPECT_RELATIVE_NEAR(value, vals[j], NegligibleValueThreshold)
//                         ++local_validated_values;
//                         break;
//                     }
//                 }
//                 break;
//             }
//         }
//     }

//     // Checking that all the values has been validated
//     const auto& r_comm = rA.Comm();
//     int global_validated_values;
//     r_comm.SumAll(&local_validated_values, &global_validated_values, 1);
//     KRATOS_EXPECT_EQ(global_validated_values, static_cast<int>(rValues.size()));
// }

// /***********************************************************************************/
// /***********************************************************************************/

// void TrilinosCPPTestExperimentalUtilities::GenerateSparseMatrixIndexAndValuesVectors(
//     const TrilinosSparseSpaceType::MatrixType& rA,
//     std::vector<int>& rRowIndexes,
//     std::vector<int>& rColumnIndexes,
//     std::vector<double>& rValues,
//     const bool PrintValues,
//     const double ThresholdIncludeHardZeros
//     )
// {
//     KRATOS_ERROR_IF_NOT(rA.Comm().NumProc() == 1) << "Debug must be done with one MPI core" << std::endl;

//     // If print values
//     if (PrintValues) {
//         std::cout << "\n        KRATOS_EXPECT_EQ(rA.NumGlobalRows(), " << rA.NumGlobalRows() << ");\n";
//         std::cout << "        KRATOS_EXPECT_EQ(rA.NumGlobalCols(), " << rA.NumGlobalCols() << ");\n";
//         std::cout << "        KRATOS_EXPECT_EQ(rA.NumGlobalNonzeros(), " << rA.NumGlobalNonzeros() << ");\n";
//     }

//     std::vector<double> values;
//     for (int i = 0; i < rA.NumMyRows(); i++) {
//         int numEntries; // Number of non-zero entries
//         double* vals;   // Row non-zero values
//         int* cols;      // Column indices of row non-zero values
//         rA.ExtractMyRowView(i, numEntries, vals, cols);
//         const int row_gid = rA.RowMap().GID(i);
//         int j;
//         for (j = 0; j < numEntries; j++) {
//             const int col_gid = rA.ColMap().GID(cols[j]);
//             if (std::abs(vals[j]) > ThresholdIncludeHardZeros) {
//                 rRowIndexes.push_back(row_gid);
//                 rColumnIndexes.push_back(col_gid);
//                 rValues.push_back(vals[j]);
//             }
//         }
//     }
//     // If print values
//     if (PrintValues) {
//         std::cout << "\n        // Values to check\n";
//         std::cout << "        std::vector<int> row_indexes = {";
//         for(std::size_t i = 0; i < rRowIndexes.size() - 1; ++i) {
//             std::cout << rRowIndexes[i] << ", ";
//         }
//         std::cout << rRowIndexes[rRowIndexes.size() - 1] << "};";
//         std::cout << "\n        std::vector<int> column_indexes = {";
//         for(std::size_t i = 0; i < rColumnIndexes.size() - 1; ++i) {
//             std::cout << rColumnIndexes[i] << ", ";
//         }
//         std::cout << rColumnIndexes[rColumnIndexes.size() - 1] << "};";
//         std::cout << "\n        std::vector<double> values = {";
//         for(std::size_t i = 0; i < rValues.size() - 1; ++i) {
//             std::cout << std::fixed;
//             std::cout << std::setprecision(16);
//             std::cout << rValues[i] << ", ";
//         }
//         std::cout << std::fixed;
//         std::cout << std::setprecision(16);
//         std::cout << rValues[rValues.size() - 1] << "};" << std::endl;
//     }
// }

// /***********************************************************************************/
// /***********************************************************************************/

// TrilinosCPPTestExperimentalUtilities::TrilinosSparseMatrixType TrilinosCPPTestExperimentalUtilities::GenerateSparseMatrix(
//     const DataCommunicator& rDataCommunicator,
//     const int NumGlobalElements,
//     const std::vector<int>& rRowIndexes,
//     const std::vector<int>& rColumnIndexes,
//     const std::vector<double>& rValues,
//     const Epetra_Map* pMap
//     )
// {
//     // Generate Epetra communicator
//     KRATOS_ERROR_IF_NOT(rDataCommunicator.IsDistributed()) << "Only distributed DataCommunicators can be used!" << std::endl;
//     auto raw_mpi_comm = MPIDataCommunicator::GetMPICommunicator(rDataCommunicator);
//     Epetra_MpiComm epetra_comm(raw_mpi_comm);

//     // Create a map
//     Epetra_Map map = (pMap == nullptr) ? Epetra_Map(NumGlobalElements,0,epetra_comm) : *pMap;

//     // Local number of rows
//     const int NumMyElements = map.NumMyElements();

//     // Get update list
//     int* MyGlobalElements = map.MyGlobalElements();

//     // Create an integer vector NumNz that is used to build the EPetra Matrix.
//     const int size_global_vector = rRowIndexes.size();
//     std::vector<int> NumNz(NumMyElements, 0);
//     int current_row_index;
//     for (current_row_index=0; current_row_index<size_global_vector; ++current_row_index) {
//         if (MyGlobalElements[0] == rRowIndexes[current_row_index]) {
//             break;
//         }
//     }
//     int current_id = rRowIndexes[current_row_index];
//     int nnz = 0;
//     int initial_index, end_index;
//     std::unordered_map<int, std::pair<int, int>> initial_and_end_index;
//     for (int i=0; i<NumMyElements; i++) {
//         if (MyGlobalElements[i] == rRowIndexes[current_row_index]) {
//             initial_index = current_row_index;
//             const int start_index = current_row_index;
//             for (current_row_index = start_index; current_row_index < size_global_vector; ++current_row_index) {
//                 if (current_id == rRowIndexes[current_row_index]) {
//                     ++nnz;
//                 } else {
//                     current_id = rRowIndexes[current_row_index];
//                     break;
//                 }
//             }
//             end_index = current_row_index;
//             initial_and_end_index.insert({i, std::make_pair(initial_index, end_index)});
//             NumNz[i] = nnz;
//             nnz = 0;
//         }
//     }

//     // Create an Epetra_Matrix
//     TrilinosSparseMatrixType A(Copy, map, NumNz.data());

//     // Fill matrix
//     int ierr;
//     auto it_end = initial_and_end_index.end();
//     auto it_index_begin = rColumnIndexes.begin();
//     auto it_values_begin = rValues.begin();
//     for( int i=0 ; i<NumMyElements; ++i ) {
//         auto it_find = initial_and_end_index.find(i);
//         if (it_find != it_end) {
//             const auto& r_pair = it_find->second;
//             initial_index = r_pair.first;
//             end_index = r_pair.second;
//             std::vector<int> indexes(it_index_begin + initial_index, it_index_begin + end_index);
//             std::vector<double> values(it_values_begin + initial_index, it_values_begin + end_index);
//             ierr = A.InsertGlobalValues(MyGlobalElements[i], end_index - initial_index, values.data(), indexes.data());
//             KRATOS_ERROR_IF_NOT(ierr == 0) << "Error in inserting values " << ierr << std::endl;
//         }
//     }

//     // Finish up, trasforming the matrix entries into local numbering,
//     // to optimize data transfert during matrix-vector products
//     ierr = A.FillComplete();
//     KRATOS_ERROR_IF_NOT(ierr == 0) << "Error in global assembling " << ierr << std::endl;

//     return A;
// }

} /// namespace Kratos