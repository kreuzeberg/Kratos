//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

// System includes

// External includes

// Project includes
#include "custom_utilities/data_transfer_3D_1D_utilities.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"
#include "utilities/atomic_utilities.h"
#include "utilities/intersection_utilities.h"
#include "utilities/variable_utils.h"
#include "spatial_containers/spatial_containers.h" // kd-tree
#include "factories/mapper_factory.h" // The mappers

namespace Kratos
{

void DataTransfer3D1DUtilities::From3Dto1DDataTransfer(
    ModelPart& rModelPart3D,
    ModelPart& rModelPart1D,
    Parameters ThisParameters
    )
{

    // Validate deafult parameters
    ThisParameters.ValidateAndAssignDefaults(GetDefaultParameters());

    // Getting variables
    std::vector<const Variable<double>*> origin_list_variables, destination_list_variables;
    GetVariablesList(ThisParameters, origin_list_variables, destination_list_variables);
    
    // Max length of the elements considered
    const double max_length = GetMaxLength(rModelPart3D);

    // The elements array
    auto& r_elements_array = rModelPart3D.Elements();
    const auto it_elem_begin = r_elements_array.begin();

    /// Type definitions for the tree
    using PointType = PointElement;
    using PointTypePointer = PointType::Pointer;
    using PointVector = std::vector<PointTypePointer>;
    using PointIterator = PointVector::iterator;
    using DistanceVector = std::vector<double>;
    using DistanceIterator = DistanceVector::iterator;

    /// KDtree definitions
    using BucketType = Bucket< 3ul, PointType, PointVector, PointTypePointer, PointIterator, DistanceIterator >;
    using KDTree = Tree< KDTreePartition<BucketType> >;

    // Some auxiliary values
    const IndexType allocation_size = ThisParameters["search_parameters"]["allocation_size"].GetInt();                 // Allocation size for the vectors and max number of potential results
    const double search_factor = ThisParameters["search_parameters"]["search_factor"].GetDouble();                     // The search factor to be considered
    const double search_increment_factor = ThisParameters["search_parameters"]["search_increment_factor"].GetDouble(); // The search increment factor to be considered
    IndexType bucket_size = ThisParameters["search_parameters"]["bucket_size"].GetInt();                               // Bucket size for kd-tree

    PointVector points_vector;
    points_vector.reserve(r_elements_array.size());
    for (std::size_t i = 0; i < r_elements_array.size(); ++i) {
        auto it_elem = it_elem_begin + i;
        points_vector.push_back(PointTypePointer(new PointType(*(it_elem.base()))));
    }
    KDTree tree_points(points_vector.begin(), points_vector.end(), bucket_size);

    // Swap sign
    const bool swap_sign = ThisParameters["swap_sign"].GetBool();
    const double constant = swap_sign ? -1.0 : 1.0;

    // Iterate over the nodes
    block_for_each(rModelPart1D.Nodes(), [&](NodeType& rNode) {
        double search_radius = search_factor * max_length;

        // Initialize values
        PointVector points_found(allocation_size);
        IndexType number_points_found = 0;
        while (number_points_found == 0) {
            search_radius *= search_increment_factor;
            const PointElement point(rNode.X(), rNode.Y(), rNode.Z());
            number_points_found = tree_points.SearchInRadius(point, search_radius, points_found.begin(), allocation_size);
        }

        array_1d<double, 3> aux_coordinates;
        Vector N, values_origin;
        for (IndexType i = 0; i < number_points_found; ++i) {
            auto p_point = points_found[i];
            auto& r_geometry = p_point->pGetElement()->GetGeometry();
            if (r_geometry.IsInside(rNode.Coordinates(), aux_coordinates)) {
                r_geometry.ShapeFunctionsValues( N, aux_coordinates );
                for (std::size_t i_var = 0; i_var < origin_list_variables.size(); ++i_var) {
                    double& r_destination_value = rNode.FastGetSolutionStepValue(*destination_list_variables[i_var]);
                    if (values_origin.size() != r_geometry.size()) {
                        values_origin.resize(r_geometry.size(), false);
                    }
                    for (std::size_t i_node = 0; i_node < r_geometry.PointsNumber(); ++i_node) {
                        auto& r_origin_node = r_geometry[i_node];
                        values_origin[i_node]= r_origin_node.FastGetSolutionStepValue(*origin_list_variables[i_var]);
                    }
                    const double origin_value = inner_prod(N, values_origin);
                    r_destination_value = constant * origin_value;
                }
            }
        }
    });
}

/***********************************************************************************/
/***********************************************************************************/

void DataTransfer3D1DUtilities::From1Dto3DDataTransfer(
    ModelPart& rModelPart3D,
    ModelPart& rModelPart1D,
    Parameters ThisParameters
    )
{
    // Validate deafult parameters
    ThisParameters.ValidateAndAssignDefaults(GetDefaultParameters());

    // Getting variables
    std::vector<const Variable<double>*> origin_list_variables, destination_list_variables;
    GetVariablesList(ThisParameters, origin_list_variables, destination_list_variables);
    
    // Max length of the elements considered
    const double max_length = GetMaxLength(rModelPart1D);

    // The elements array
    auto& r_elements_array = rModelPart1D.Elements();
    const auto it_elem_begin = r_elements_array.begin();

    /// Type definitions for the tree
    using PointType = PointElement;
    using PointTypePointer = PointType::Pointer;
    using PointVector = std::vector<PointTypePointer>;
    using PointIterator = PointVector::iterator;
    using DistanceVector = std::vector<double>;
    using DistanceIterator = DistanceVector::iterator;

    /// KDtree definitions
    using BucketType = Bucket< 3ul, PointType, PointVector, PointTypePointer, PointIterator, DistanceIterator >;
    using KDTree = Tree< KDTreePartition<BucketType> >;

    // Some auxiliary values
    const double tolerance = 1.0e-16;
    const IndexType allocation_size = ThisParameters["search_parameters"]["allocation_size"].GetInt();                 // Allocation size for the vectors and max number of potential results
    const double search_factor = ThisParameters["search_parameters"]["search_factor"].GetDouble();                     // The search factor to be considered
    const double search_increment_factor = ThisParameters["search_parameters"]["search_increment_factor"].GetDouble(); // The search increment factor to be considered
    IndexType bucket_size = ThisParameters["search_parameters"]["bucket_size"].GetInt();                               // Bucket size for kd-tree

    PointVector points_vector;
    points_vector.reserve(r_elements_array.size());
    for (std::size_t i = 0; i < r_elements_array.size(); ++i) {
        auto it_elem = it_elem_begin + i;
        points_vector.push_back(PointTypePointer(new PointType(*(it_elem.base()))));
    }
    KDTree tree_points(points_vector.begin(), points_vector.end(), bucket_size);

    // Auxiliary values
    struct AuxValues {
        array_1d<double, 3> aux_coordinates;
        array_1d<double,3> intersection_point1, intersection_point2;
        array_1d<double,2> values_origin, values_origin_interpolated;
        array_1d<double,4> values_destination;
        BoundedMatrix<double, 2, 4> N_values;
        BoundedMatrix<double, 4, 2> inverted_N_values;
        std::array<GeometryType::Pointer, 2> lines;
        std::array<array_1d<double, 3>, 2> line_points;
        std::array<Vector, 4> N_line;
        Vector N_tetra = ZeroVector(4);
        double aux_det;
    };

    // Initialize the NODAL_VOLUME
    VariableUtils().SetNonHistoricalVariableToZero(NODAL_VOLUME, rModelPart3D.Nodes());

    // Debug mode
    const bool debug_mode = ThisParameters["debug_mode"].GetBool();

    // Iterate over the elements (first assign NODAL_VOLUME)
    block_for_each(rModelPart3D.Elements(), AuxValues(), [&](Element& rElement, AuxValues av) {
        double search_radius = search_factor * max_length;

        // Initialize values
        PointVector points_found(allocation_size);
        IndexType number_points_found = 0;
        auto& r_geometry_tetra = rElement.GetGeometry();

        // Iterate doing the search
        const Point center = r_geometry_tetra.Center();
        while (number_points_found == 0) {
            search_radius *= search_increment_factor;
            const PointElement point(center.X(), center.Y(), center.Z());
            number_points_found = tree_points.SearchInRadius(point, search_radius, points_found.begin(), allocation_size);
        }

        // Getting the intersection points
        unsigned int counter = 0;
        for (IndexType i = 0; i < number_points_found; ++i) {
            auto p_point = points_found[i];
            auto p_geometry = p_point->pGetElement()->pGetGeometry();
            auto& r_geometry = *p_geometry;
            const int intersection = IntersectionUtilities::ComputeTetrahedraLineIntersection<GeometryType, false>(r_geometry_tetra, r_geometry[0].Coordinates(), r_geometry[1].Coordinates(), av.intersection_point1, av.intersection_point2);
            if (intersection == 1) { // Two intersection points
                counter += 2;
            } else if (intersection == 2) { // Two points, one inside the tetrahedra and the other outside
                bool add_point = true;
                for (unsigned int i_point = 0; i_point < counter; ++i_point) {
                    if (norm_2(av.line_points[i_point] - av.intersection_point1) < tolerance) {
                        add_point = false;
                        break;
                    }
                }
                if (add_point) {
                    counter += 1;
                }
            }
            if (counter > 1) {
                break;
            }
        }
        // 2 points are required
        if (counter > 1) {
            // For debugging purposes
            if (debug_mode) rElement.Set(VISITED); 

            // Getting volume
            const double volume = r_geometry_tetra.Volume();

            // Iterate over the nodes of the element and add the contribution of the volume
            for (unsigned int i_node = 0; i_node < 4; ++i_node) {
                auto& r_value = r_geometry_tetra[i_node].GetValue(NODAL_VOLUME);
                AtomicAdd(r_value, volume);
            }
        }
    });

    // Initialize if required
    block_for_each(rModelPart3D.Nodes(), [&destination_list_variables](auto& rNode) {
        const double nodal_volume = rNode.GetValue(NODAL_VOLUME);
        if (nodal_volume > std::numeric_limits<double>::epsilon()) {
            for (std::size_t i_var = 0; i_var < destination_list_variables.size(); ++i_var) {
                rNode.FastGetSolutionStepValue(*destination_list_variables[i_var]) = 0.0;
            }
        }
    });

    // Swap sign
    const bool swap_sign = ThisParameters["swap_sign"].GetBool();
    const double constant = swap_sign ? -1.0 : 1.0;

    // Iterate over the elements
    block_for_each(rModelPart3D.Elements(), AuxValues(), [&](Element& rElement, AuxValues av) {
        double search_radius = search_factor * max_length;

        // Initialize values
        PointVector points_found(allocation_size);
        IndexType number_points_found = 0;
        auto& r_geometry_tetra = rElement.GetGeometry();

        // Getting volume
        const double volume = r_geometry_tetra.Volume();

        // Iterate doing the search
        const Point center = r_geometry_tetra.Center();
        while (number_points_found == 0) {
            search_radius *= search_increment_factor;
            const PointElement point(center.X(), center.Y(), center.Z());
            number_points_found = tree_points.SearchInRadius(point, search_radius, points_found.begin(), allocation_size);
        }

        // Getting the intersection points
        unsigned int counter = 0;
        for (IndexType i = 0; i < number_points_found; ++i) {
            auto p_point = points_found[i];
            auto p_geometry = p_point->pGetElement()->pGetGeometry();
            auto& r_geometry = *p_geometry;
            const int intersection = IntersectionUtilities::ComputeTetrahedraLineIntersection<GeometryType, false>(r_geometry_tetra, r_geometry[0].Coordinates(), r_geometry[1].Coordinates(), av.intersection_point1, av.intersection_point2);
            // TODO: Actually a better alternative could be to do "a mortar", considering mass matrices between the line and the tetrahedra. This requires some extra work, and I think that the current implementation is enough for the moment
            if (intersection == 1) { // Two intersection points
                av.line_points[0] = av.intersection_point1;
                av.line_points[1] = av.intersection_point2;
                for (unsigned int i_line = 0; i_line < 2; ++i_line) {
                    av.lines[i_line] = p_geometry;
                }
                counter += 2;
            } else if (intersection == 2) { // One intersection point
                bool add_point = true;
                if (counter == 1) {
                    if (norm_2(av.line_points[0] - av.intersection_point1) < tolerance) {
                        add_point = false;
                        break;
                    }
                }
                if (add_point) {
                    av.lines[counter] = p_geometry;
                    av.line_points[counter] = av.intersection_point1;
                    counter += 1;
                }
            } else if (intersection > 4) { // One intersection point in a corner
                // Detect the node of the tetrahedra and directly assign
                const int index_node = intersection - 5;
                // Assign value
                auto& r_node = r_geometry_tetra[index_node];
                p_geometry->PointLocalCoordinates(av.aux_coordinates, av.intersection_point1);
                p_geometry->ShapeFunctionsValues( av.N_line[0], av.aux_coordinates );
                for (std::size_t i_var = 0; i_var < origin_list_variables.size(); ++i_var) {
                    av.values_origin[0] = r_geometry[0].FastGetSolutionStepValue(*origin_list_variables[i_var]);
                    av.values_origin[1] = r_geometry[1].FastGetSolutionStepValue(*origin_list_variables[i_var]);
                    r_node.FastGetSolutionStepValue(*destination_list_variables[i_var]) = MathUtils<double>::Dot(av.N_line[0], av.values_origin);
                }
                break;
            } 
            if (counter > 1) {
                break;
            }
        }
        // 2 points are required
        if (counter > 1) {
            // Common operation
            for (unsigned int i_line = 0; i_line < 2; ++i_line) {
                av.lines[i_line]->PointLocalCoordinates(av.aux_coordinates, av.line_points[i_line]);
                av.lines[i_line]->ShapeFunctionsValues( av.N_line[i_line], av.aux_coordinates );
            }

            // Fill the N matrix
            for (unsigned int i = 0; i < 2; ++i) {
                r_geometry_tetra.PointLocalCoordinates(av.aux_coordinates, av.line_points[i]);
                r_geometry_tetra.ShapeFunctionsValues( av.N_tetra, av.aux_coordinates );
                for (unsigned int j = 0; j < 4; ++j) {
                    av.N_values(i, j) = av.N_tetra[j];
                }
            }

            // Calculate values
            MathUtils<double>::GeneralizedInvertMatrix( av.N_values, av.inverted_N_values, av.aux_det);
            for (std::size_t i_var = 0; i_var < origin_list_variables.size(); ++i_var) {
                for (unsigned int i_line = 0; i_line < 2; ++i_line) {
                    auto& r_line = *av.lines[i_line];
                    av.values_origin[0] = r_line[0].FastGetSolutionStepValue(*origin_list_variables[i_var]);
                    av.values_origin[1] = r_line[1].FastGetSolutionStepValue(*origin_list_variables[i_var]);
                    av.values_origin_interpolated[i_line] = MathUtils<double>::Dot(av.N_line[i_line], av.values_origin);
                }
                noalias(av.values_destination) = prod(av.inverted_N_values, av.values_origin_interpolated);
                for (unsigned int i_node = 0; i_node < 4; ++i_node) {
                    auto& r_value = r_geometry_tetra[i_node].FastGetSolutionStepValue(*destination_list_variables[i_var]);
                    AtomicAdd(r_value, constant * av.values_destination[i_node] * volume);
                }
            }
        }
    });

    // Normalize by the NODAL_VOLUME
    block_for_each(rModelPart3D.Nodes(), [&destination_list_variables](auto& rNode) {
        const double nodal_volume = rNode.GetValue(NODAL_VOLUME);
        if (nodal_volume > std::numeric_limits<double>::epsilon()) {
            for (std::size_t i_var = 0; i_var < destination_list_variables.size(); ++i_var) {
                rNode.FastGetSolutionStepValue(*destination_list_variables[i_var]) /= nodal_volume;
            }
        }
    });

    // Remove the NODAL_VOLUME
    VariableUtils().EraseNonHistoricalVariable(NODAL_VOLUME, rModelPart3D.Nodes());
}

/***********************************************************************************/
/***********************************************************************************/

void DataTransfer3D1DUtilities::GetVariablesList(
    Parameters ThisParameters,
    std::vector<const Variable<double>*>& rOriginListVariables,
    std::vector<const Variable<double>*>& rDestinationListVariables
    )
{
    // The components as an array of strings
    const std::array<std::string, 3> direction_string({"X", "Y", "Z"});

    // Getting variables
    const std::vector<std::string> origin_variables_names = ThisParameters["origin_variables"].GetStringArray();
    const std::vector<std::string> destination_variables_names = ThisParameters["destination_variables"].GetStringArray();
    KRATOS_ERROR_IF(origin_variables_names.size() == 0) << "No variables defined" << std::endl;
    KRATOS_ERROR_IF(origin_variables_names.size() != destination_variables_names.size()) << "Origin and destination variables do not coincide in size" << std::endl;
    for (IndexType i_var = 0; i_var < origin_variables_names.size(); ++i_var) {
        if (KratosComponents<Variable<double>>::Has(origin_variables_names[i_var])) {
            rOriginListVariables.push_back(&KratosComponents<Variable<double>>::Get(origin_variables_names[i_var]));
            rDestinationListVariables.push_back(&KratosComponents<Variable<double>>::Get(destination_variables_names[i_var]));
        } else if (KratosComponents<Variable<array_1d<double, 3>>>::Has(origin_variables_names[i_var])) {
            const auto& r_origin_var_name = origin_variables_names[i_var];
            const auto& r_destination_var_name = destination_variables_names[i_var];
            for (unsigned int i_comp = 0; i_comp < 3; ++i_comp) {
                rOriginListVariables.push_back(&KratosComponents<Variable<double>>::Get(r_origin_var_name + "_" + direction_string[i_comp]));
                rDestinationListVariables.push_back(&KratosComponents<Variable<double>>::Get(r_destination_var_name + "_" + direction_string[i_comp]));
            }
        } else {
            KRATOS_ERROR << "Variable " << origin_variables_names[i_var] << " not available in KratosComponents as double or 3 components array" << std::endl;
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

void DataTransfer3D1DUtilities::InterpolateFrom1Dto3D(
    ModelPart& rModelPart3D,
    ModelPart& rModelPart1D,
    Parameters ThisParameters
    )
{
    
}

/***********************************************************************************/
/***********************************************************************************/

void DataTransfer3D1DUtilities::InterpolateFrom3Dto1D(
    ModelPart& rModelPart3D,
    ModelPart& rModelPart1D,
    Parameters ThisParameters
    )
{
    
}

/***********************************************************************************/
/***********************************************************************************/

void DataTransfer3D1DUtilities::ExtrapolateFrom1Dto3D(
    ModelPart& rModelPart3D,
    ModelPart& rModelPart1D,
    Parameters ThisParameters
    )
{
    
}

/***********************************************************************************/
/***********************************************************************************/

void DataTransfer3D1DUtilities::ExtrapolateFrom3Dto1D(
    ModelPart& rModelPart3D,
    ModelPart& rModelPart1D,
    Parameters ThisParameters
    )
{
    
}

/***********************************************************************************/
/***********************************************************************************/

double DataTransfer3D1DUtilities::GetMaxLength(ModelPart& rModelPart)
{
    auto& r_elements_array = rModelPart.Elements();
    KRATOS_ERROR_IF(r_elements_array.size() == 0) << "Empty model part" << std::endl;
    return block_for_each<MaxReduction<double>>(r_elements_array, [&](Element& rElement) {
        return rElement.GetGeometry().Length();
    });
}

/***********************************************************************************/
/***********************************************************************************/

Parameters DataTransfer3D1DUtilities::GetDefaultParameters()
{
    Parameters default_parameters = Parameters(R"(
    {
        "origin_variables"         : [],
        "destination_variables"    : [],
        "swap_sign"                : false,
        "debug_mode"               : false,
        "extrapolate_values"       : false,
        "search_parameters"        :  {
            "allocation_size"         : 100,
            "bucket_size"             : 4,
            "search_factor"           : 2.0,
            "search_increment_factor" : 1.5
        }
    })" );

    return default_parameters;
}

} // namespace Kratos