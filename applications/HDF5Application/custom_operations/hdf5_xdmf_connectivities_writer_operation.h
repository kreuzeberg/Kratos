//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:        BSD License
//                  license: HDF5Application/license.txt
//
//  Main author:    Michael Andre, https://github.com/msandre
//                  Suneht Warnakulasuriya
//

#pragma once

// System includes
#include <string>
#include <vector>
#include <unordered_map>

// External includes

// Project includes
#include "includes/define.h"
#include "operations/operation.h"

// Application includes
#include "hdf5_application_define.h"
#include "custom_io/hdf5_file.h"

namespace Kratos
{
namespace HDF5
{
///@addtogroup HDF5Application
///@{
///@name Kratos Classes
///@{

/// Writes Xdmf connectivities.
/**
 * In partitioned simulations, the node ids are not sorted globally. Thus
 * the Kratos element connectivities cannot be used directly to index the
 * node arrays in the HDF5 file. This utility transforms Kratos element
 * connectivities to Xdmf connectivities, which can be used directly to
 * index the node arrays.
 */
class KRATOS_API(HDF5_APPLICATION) XdmfConnectivitiesWriterOperation : public Operation
{
public:
    ///@name Type Definitions
    ///@{

    using IdMapType = std::unordered_map<int, int>;

    /// Pointer definition
    KRATOS_CLASS_POINTER_DEFINITION(XdmfConnectivitiesWriterOperation);

    ///@}
    ///@name Life Cycle
    ///@{

    XdmfConnectivitiesWriterOperation(const std::string& rFileName);

    ///@}
    ///@name Operations
    ///@{

    void Execute() override;

    ///@}

private:
    ///@name Member Variables
    ///@{

    DataCommunicator mSerialDataCommunicator;

    File::Pointer mpFile;

    std::vector<std::string> mListOfModelDataPaths;

    ///@}
    ///@name Private Operations
    ///@{

    void CreateXdmfModelData(
        const std::string& rModelDataPath) const;

    void CreateXdmfPoints(
        const std::string& rKratosNodeIdsPath,
        const std::string& rXdmfNodeIdsPath,
        const IdMapType& rKratosToXdmfIdMap) const;

    void CreateXdmfConnectivities(
        const std::string& rKratosConnectivitiesPath,
        const std::string& rXdmfConnectivitiesPath,
        const IdMapType& rKratosToXdmfIdMap) const;

    void CreateXdmfConnectivitiesForSubModelParts(
        const std::string& rPath,
        const std::string& rDestinationPrefix,
        const IdMapType& rKratosToXdmfIdMap) const;

    ///@}

}; // class XdmfConnectivitiesWriterOperation

///@} // Kratos Classes
///@} addtogroup
} // namespace HDF5.
} // namespace Kratos.