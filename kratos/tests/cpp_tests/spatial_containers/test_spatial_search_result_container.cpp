//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//
//

// System includes

// External includes

// Project includes
#include "testing/testing.h"
#include "geometries/line_2d_2.h"
#include "spatial_containers/spatial_search_result.h"
#include "spatial_containers/spatial_search_result_container.h"

namespace Kratos::Testing 
{

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerAddResult, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Create a test result
    GeometricalObject object = GeometricalObject(1);
    SpatialSearchResult<GeometricalObject> result(&object);
    result.SetDistance(0.5);

    // Add the result to the container
    container.AddResult(result);

    // Check that the result was added correctly
    auto& r_local_pointers = container.GetLocalResults();
    KRATOS_EXPECT_EQ(r_local_pointers.size(), 1);
    KRATOS_EXPECT_EQ(r_local_pointers.size(), container.NumberOfLocalResults());

    // Check distances
    KRATOS_EXPECT_EQ(container[0].GetDistance(), 0.5);

    // Check global pointers
    KRATOS_EXPECT_FALSE(container.IsObjectFound());
    auto& r_global_pointers = container.GetGlobalResults();
    KRATOS_EXPECT_EQ(r_global_pointers.size(), 0); // It should be empty as we have not synchronized
    KRATOS_EXPECT_EQ(r_global_pointers.size(), container.NumberOfGlobalResults()); // It should be empty as we have not synchronized
}

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerClear, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Create a test result
    GeometricalObject object = GeometricalObject(1);
    SpatialSearchResult<GeometricalObject> result(&object);
    result.SetDistance(0.5);

    // Add the result to the container
    container.AddResult(result);

    // Clear
    container.Clear();

    // Check that the result was added correctly
    auto& r_local_pointers = container.GetLocalResults();
    KRATOS_EXPECT_EQ(r_local_pointers.size(), 0);
    KRATOS_EXPECT_EQ(r_local_pointers.size(), container.NumberOfLocalResults());
}

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerSynchronizeAll, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Create a test result
    GeometricalObject object = GeometricalObject(1);
    SpatialSearchResult<GeometricalObject> result(&object);

    // Add the result to the container
    container.AddResult(result);

    // Synchronize the container between partitions
    DataCommunicator data_communicator;
    container.SynchronizeAll(data_communicator);

    // Check that the result was added correctly
    auto& r_local_pointers = container.GetLocalResults();
    KRATOS_EXPECT_EQ(r_local_pointers.size(), 1);
    KRATOS_EXPECT_EQ(r_local_pointers.size(), container.NumberOfLocalResults());

    // Check global pointers
    KRATOS_EXPECT_TRUE(container.IsObjectFound());
    auto& r_global_pointers = container.GetGlobalResults();
    KRATOS_EXPECT_EQ(r_global_pointers.size(), 1);
    KRATOS_EXPECT_EQ(r_global_pointers.size(), container.NumberOfGlobalResults());
}

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerGetResultShapeFunctions, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Generate a geometry
    auto p_node1 = Kratos::make_intrusive<Node>(1, 0.0, 0.0, 0.0);
    auto p_node2 = Kratos::make_intrusive<Node>(2, 1.0, 0.0, 0.0);
    Geometry<Node>::Pointer p_geom = Kratos::make_shared<Line2D2<Node>>(p_node1, p_node2);

    // Create a test result
    GeometricalObject object = GeometricalObject(1, p_geom);
    SpatialSearchResult<GeometricalObject> result(&object);

    // Add the result to the container
    container.AddResult(result);

    // Synchronize the container between partitions
    DataCommunicator data_communicator;
    container.SynchronizeAll(data_communicator);

    // Compute shape functions
    Point point = Point(0.5, 0.0, 0.0);
    auto shape_functions = container.GetResultShapeFunctions(point);

    // Check shape functions
    KRATOS_EXPECT_EQ(shape_functions.size(), 1);
    KRATOS_EXPECT_NEAR(shape_functions[0][0], 0.5, 1.0e-12);
    KRATOS_EXPECT_NEAR(shape_functions[0][1], 0.5, 1.0e-12);
}

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerGetResultIndices, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Create a test result
    GeometricalObject object = GeometricalObject(1);
    SpatialSearchResult<GeometricalObject> result(&object);

    // Add the result to the container
    container.AddResult(result);

    // Synchronize the container between partitions
    DataCommunicator data_communicator;
    container.SynchronizeAll(data_communicator);

    // Compute indices
    auto indices = container.GetResultIndices();

    // Check indices
    KRATOS_EXPECT_EQ(indices.size(), 1);
    KRATOS_EXPECT_EQ(indices[0], object.Id());
}

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerGetResultNodeIndices, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Generate a geometry
    auto p_node1 = Kratos::make_intrusive<Node>(1, 0.0, 0.0, 0.0);
    auto p_node2 = Kratos::make_intrusive<Node>(2, 1.0, 0.0, 0.0);
    Geometry<Node>::Pointer p_geom = Kratos::make_shared<Line2D2<Node>>(p_node1, p_node2);

    // Create a test result
    GeometricalObject object = GeometricalObject(1, p_geom);
    SpatialSearchResult<GeometricalObject> result(&object);

    // Add the result to the container
    container.AddResult(result);

    // Synchronize the container between partitions
    DataCommunicator data_communicator;
    container.SynchronizeAll(data_communicator);

    // Compute indices
    auto indices = container.GetResultNodeIndices();

    // Check indices
    KRATOS_EXPECT_EQ(indices.size(), 1);
    KRATOS_EXPECT_EQ(indices[0][0], 1);
    KRATOS_EXPECT_EQ(indices[0][1], 2);
}

KRATOS_TEST_CASE_IN_SUITE(SpatialSearchResultContainerGetResultCoordinates, KratosCoreFastSuite)
{
    // Create a test object
    SpatialSearchResultContainer<GeometricalObject> container;

    // Generate a geometry
    auto p_node1 = Kratos::make_intrusive<Node>(1, 0.0, 0.0, 0.0);
    auto p_node2 = Kratos::make_intrusive<Node>(2, 1.0, 0.0, 0.0);
    Geometry<Node>::Pointer p_geom = Kratos::make_shared<Line2D2<Node>>(p_node1, p_node2);

    // Create a test result
    GeometricalObject object = GeometricalObject(1, p_geom);
    SpatialSearchResult<GeometricalObject> result(&object);

    // Add the result to the container
    container.AddResult(result);

    // Synchronize the container between partitions
    DataCommunicator data_communicator;
    container.SynchronizeAll(data_communicator);

    // Compute result coordinates
    auto coordinates = container.GetResultCoordinates();

    // Check result coordinates
    KRATOS_EXPECT_EQ(coordinates.size(), 1);
    KRATOS_EXPECT_EQ(coordinates[0].size(), 2);
    KRATOS_EXPECT_VECTOR_NEAR(coordinates[0][0], p_node1->Coordinates(), 1.0e-12);
    KRATOS_EXPECT_VECTOR_NEAR(coordinates[0][1], p_node2->Coordinates(), 1.0e-12);
}

}  // namespace Kratos::Testing