//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Joaquin Gonzalez-Usua
//
//

//These formulas are derived from "Derivatives.py" script

// System includes

// External includes

// Project includes
#include "containers/model.h"
#include "includes/checks.h"
#include "utilities/variable_utils.h"
#include "utilities/math_utils.h"

// Application includes
#include "swimming_DEM_application.h"
#include "codina2001_porosity_solution_and_body_force_process.h"
#include "swimming_dem_application_variables.h"

// Other applications includes
#include "fluid_dynamics_application_variables.h"

namespace Kratos
{

extern DenseVector<std::vector<double>> mExactScalar;
extern DenseVector<Matrix> mExactVector;
extern DenseVector<Matrix> mExactScalarGradient;
extern DenseVector<DenseVector<Matrix>> mExactVectorGradient;
/* Public functions *******************************************************/
Codina2001PorositySolutionAndBodyForceProcess::Codina2001PorositySolutionAndBodyForceProcess(
    ModelPart& rModelPart)
    : Process(),
      mrModelPart(rModelPart)
{}

Codina2001PorositySolutionAndBodyForceProcess::Codina2001PorositySolutionAndBodyForceProcess(
    ModelPart& rModelPart,
    Parameters& rParameters)
    : Process(),
      mrModelPart(rModelPart)
{
    // Check default settings
    this->CheckDefaultsAndProcessSettings(rParameters);
}

Codina2001PorositySolutionAndBodyForceProcess::Codina2001PorositySolutionAndBodyForceProcess(
    Model &rModel,
    Parameters &rParameters)
    : Process(),
      mrModelPart(rModel.GetModelPart(rParameters["model_part_name"].GetString()))
{
    // Check default settings
    this->CheckDefaultsAndProcessSettings(rParameters);

}


void Codina2001PorositySolutionAndBodyForceProcess::CheckDefaultsAndProcessSettings(Parameters &rParameters)
{
    const Parameters default_parameters = GetDefaultParameters();

    rParameters.ValidateAndAssignDefaults(default_parameters);

    mDensity = rParameters["benchmark_parameters"]["density"].GetDouble();
    mAlpha = rParameters["benchmark_parameters"]["alpha"].GetDouble();
    mViscosity = rParameters["benchmark_parameters"]["viscosity"].GetDouble();
    mSigma = rParameters["benchmark_parameters"]["sigma"].GetDouble();
    mInitialConditions = rParameters["benchmark_parameters"]["use_initial_conditions"].GetBool();
    mAlternativeFormulation = rParameters["benchmark_parameters"]["use_alternative_formulation"].GetBool();

}

const Parameters Codina2001PorositySolutionAndBodyForceProcess::GetDefaultParameters() const
{
    const Parameters default_parameters( R"(
    {
                "model_part_name"          : "please_specify_model_part_name",
                "variable_name"            : "BODY_FORCE",
                "benchmark_name"           : "custom_body_force.vortex",
                "benchmark_parameters"     : {
                                                "velocity"    : 1.0,
                                                "viscosity"   : 0.1,
                                                "density"     : 1.0,
                                                "alpha"       : 1.0,
                                                "u_char"      : 100.0,
                                                "sigma"       : 0.0,
                                                "use_initial_conditions" : true,
                                                "use_alternative_formulation" : false
                },
                "compute_nodal_error"      : true,
                "print_convergence_output" : false,
                "output_parameters"        : {}
    }  )" );

    return default_parameters;
}

void Codina2001PorositySolutionAndBodyForceProcess::Execute()
{
    this->ExecuteInitialize();
    this->ExecuteInitializeSolutionStep();
}

void Codina2001PorositySolutionAndBodyForceProcess::ExecuteInitialize()
{}

void Codina2001PorositySolutionAndBodyForceProcess::ExecuteBeforeSolutionLoop()
{
    //this->SetFluidProperties();
    if (mInitialConditions == true)
    {
        this->SetInitialBodyForceAndPorosityField();
        this->SetValuesOnIntegrationPoints();
    }
}

void Codina2001PorositySolutionAndBodyForceProcess::ExecuteInitializeSolutionStep()
{}

void Codina2001PorositySolutionAndBodyForceProcess::ExecuteFinalizeSolutionStep() {}

/* Protected functions ****************************************************/

void Codina2001PorositySolutionAndBodyForceProcess::SetInitialBodyForceAndPorosityField()
{
    double Dim = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];
    const double alpha = mAlpha;
    const double rho = mDensity;
    const double nu = mViscosity;
    Matrix I = IdentityMatrix(Dim, Dim);

    double du1dt, du2dt, du11, du12, du111, du112, du121, du122, du21, du22, du211, du212, du221, du222;

    // Computation of the BodyForce and Porosity fields
    for (auto it_node = mrModelPart.NodesBegin(); it_node != mrModelPart.NodesEnd(); it_node++){

        const double x1 = it_node->X();
        const double x2 = it_node->Y();

        double& r_mass_source = it_node->FastGetSolutionStepValue(MASS_SOURCE);

        double& r_alpha = it_node->FastGetSolutionStepValue(FLUID_FRACTION);
        double& r_dalphat = it_node->FastGetSolutionStepValue(FLUID_FRACTION_RATE);

        double& r_alpha1 = it_node->FastGetSolutionStepValue(FLUID_FRACTION_GRADIENT_X);
        double& r_alpha2 = it_node->FastGetSolutionStepValue(FLUID_FRACTION_GRADIENT_Y);

        double& r_body_force1 = it_node->FastGetSolutionStepValue(BODY_FORCE_X);
        double& r_body_force2 = it_node->FastGetSolutionStepValue(BODY_FORCE_Y);

        double& r_u1 = it_node->FastGetSolutionStepValue(EXACT_VELOCITY_X);
        double& r_u2 = it_node->FastGetSolutionStepValue(EXACT_VELOCITY_Y);

        double& r_pressure = it_node->FastGetSolutionStepValue(EXACT_PRESSURE);

        Matrix& sigma = it_node->FastGetSolutionStepValue(PERMEABILITY);


		r_u1 = std::pow(x1,2)*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

        r_u2 = std::pow(x2,2)*std::pow((1 - x2),2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

        r_alpha = alpha;

        du1dt = 0.0;

        du2dt = 0.0;

        r_dalphat = 0.0;

        r_alpha1 = 0.0;

        r_alpha2 = 0.0;

        du11 = 7*std::pow(x1,2)*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + std::pow(x1,2)*(2*x1 - 2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 2*x1*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

        du12 = std::pow(x1,2)*std::pow((1 - x1),2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

        du111 = 49*std::pow(x1,2)*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 14*std::pow(x1,2)*(2*x1 - 2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 2*std::pow(x1,2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 28*x1*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 4*x1*(2*x1 - 2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 2*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

        du112 = 7*std::pow(x1,2)*std::pow((1 - x1),2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + std::pow(x1,2)*(2*x1 - 2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 2*x1*std::pow((1 - x1),2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

        du121 = 7*std::pow(x1,2)*std::pow((1 - x1),2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + std::pow(x1,2)*(2*x1 - 2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 2*x1*std::pow((1 - x1),2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

        du122 = std::pow(x1,2)*std::pow((1 - x1),2)*(24*x2 - 12)*std::exp(7*x1)/alpha;

        du21 = std::pow(x2,2)*std::pow((1 - x2),2)*(-49*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 14*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow(x1,2)*std::exp(7*x1) - 28*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 4*x1*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

        du22 = std::pow(x2,2)*(2*x2 - 2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha + 2*x2*std::pow((1 - x2),2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

        du211 = std::pow(x2,2)*std::pow((1 - x2),2)*(-343*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 147*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 42*std::pow(x1,2)*std::exp(7*x1) - 294*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 84*x1*(2*x1 - 2)*std::exp(7*x1) - 12*x1*std::exp(7*x1) - 42*std::pow((1 - x1),2)*std::exp(7*x1) + (4 - 4*x1)*std::exp(7*x1) + (8 - 8*x1)*std::exp(7*x1))/alpha;

        du212 = std::pow(x2,2)*(2*x2 - 2)*(-49*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 14*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow(x1,2)*std::exp(7*x1) - 28*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 4*x1*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow((1 - x1),2)*std::exp(7*x1))/alpha + 2*x2*std::pow((1 - x2),2)*(-49*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 14*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow(x1,2)*std::exp(7*x1) - 28*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 4*x1*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

        du221 = std::pow(x2,2)*(2*x2 - 2)*(-49*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 14*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow(x1,2)*std::exp(7*x1) - 28*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 4*x1*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow((1 - x1),2)*std::exp(7*x1))/alpha + 2*x2*std::pow((1 - x2),2)*(-49*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 14*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow(x1,2)*std::exp(7*x1) - 28*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 4*x1*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

        du222 =  2*std::pow(x2,2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha + 4*x2*(2*x2 - 2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha + 2*std::pow((1 - x2),2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

        r_pressure = 0.0;

        sigma = mSigma * I;

        const double convective1 = r_u1 * du11 + r_u2 * du12;
        const double convective2 = r_u1 * du21 + r_u2 * du22;

        const double div_of_sym_grad1 = (1.0/2.0) * (2 * du111 + du212 + du122);
        const double div_of_sym_grad2 = (1.0/2.0) * (du121 + du211 + 2 * du222);

        const double grad_of_div1 = du111 + du221;
        const double grad_of_div2 = du112 + du222;

        const double press_grad1 = 0.0;
        const double press_grad2 = 0.0;

        if (mAlternativeFormulation){

            const double grad_alpha_sym_grad1 = (1.0/2.0) * (2 * r_alpha1 * du11 + r_alpha2 * (du21 + du12));
            const double grad_alpha_sym_grad2 = (1.0/2.0) * (r_alpha1 * (du12 + du21) + 2 * r_alpha2 * du22);

            const double grad_alpha_div1 = r_alpha1 * (du11 + du22);
            const double grad_alpha_div2 = r_alpha2 * (du11 + du22);

            r_body_force1 = r_alpha * du1dt + r_alpha * convective1 + r_alpha / rho * press_grad1 - 2.0 * nu * (r_alpha * div_of_sym_grad1 + grad_alpha_sym_grad1) + (2.0/3.0) * nu * (r_alpha * grad_of_div1 + grad_alpha_div1) + sigma(0,0) * r_u1 + sigma(0,1) * r_u2;

            r_body_force2 = r_alpha * du2dt + r_alpha * convective2 + r_alpha / rho * press_grad2 - 2.0 * nu * (r_alpha * div_of_sym_grad2 + grad_alpha_sym_grad2) + (2.0/3.0) * nu * (r_alpha * grad_of_div2 + grad_alpha_div2) + sigma(1,0) * r_u1 + sigma(1,1) * r_u2;
        }else{
            r_body_force1 = du1dt + convective1 + 1.0/rho * press_grad1 - 2 * nu * div_of_sym_grad1 + (2.0/3.0) * nu * grad_of_div1 + sigma(0,0) * r_u1 + sigma(1,0) * r_u1;

            r_body_force2 = du2dt + convective2 + 1.0/rho * press_grad2 - 2 * nu * div_of_sym_grad2 + (2.0/3.0) * nu * grad_of_div2 + sigma(0,1) * r_u2 + sigma(1,1) * r_u2;
        }

        r_mass_source = (r_dalphat + r_u1 * r_alpha1 + r_u2 * r_alpha2 + r_alpha * (du11 + du22));

    }

}

void Codina2001PorositySolutionAndBodyForceProcess::SetValuesOnIntegrationPoints()
{
    double Dim = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];
    const double alpha = mAlpha;
    Matrix I = IdentityMatrix(Dim, Dim);

    double u1, u2, pressure, du11, du12, du21, du22;

    unsigned int n_elem = mrModelPart.NumberOfElements();

    // Computation of the BodyForce and Porosity fields
    for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem){

        const auto it_elem = mrModelPart.ElementsBegin() + i_elem;
        int id_elem = it_elem->Id();
        const GeometryType& r_geometry = it_elem->GetGeometry();

        const GeometryData::IntegrationMethod integration_method = it_elem->GetIntegrationMethod();
        const auto& r_number_integration_points = r_geometry.IntegrationPointsNumber(integration_method);

        Matrix NContainer = r_geometry.ShapeFunctionsValues(integration_method);
        const unsigned int NumNodes = r_geometry.PointsNumber();

        std::vector<double> pressure_on_gauss_points;
        Matrix velocity_on_gauss_points = ZeroMatrix(Dim, r_number_integration_points);
        Matrix pressure_gradient_on_gauss_points = ZeroMatrix(Dim, r_number_integration_points);
        DenseVector<Matrix> velocity_gradient_on_gauss_points(r_number_integration_points);

        if (mExactScalar.size() != n_elem)
            mExactScalar.resize(n_elem);

        if (mExactVector.size() != n_elem)
            mExactVector.resize(n_elem);

        if (mExactScalarGradient.size() != n_elem)
            mExactScalarGradient.resize(n_elem);

        if (mExactVectorGradient.size() != n_elem)
            mExactVectorGradient.resize(n_elem);

        if (pressure_on_gauss_points.size() != r_number_integration_points)
            pressure_on_gauss_points.resize(r_number_integration_points);


        for (unsigned int g = 0; g < r_number_integration_points; g++){

            Matrix gauss_point_coordinates = ZeroMatrix(r_number_integration_points,Dim);
            for (unsigned int i = 0; i < NumNodes; ++i){
                const array_1d<double, 3>& r_coordinates = r_geometry[i].Coordinates();
                for (unsigned int d = 0; d < Dim; ++d)
                    gauss_point_coordinates(g,d) += NContainer(g,i) * r_coordinates[d];
                }

            velocity_gradient_on_gauss_points[g] = ZeroMatrix(Dim,Dim);

            const double x1 = gauss_point_coordinates(g,0);
            const double x2 = gauss_point_coordinates(g,1);

		    u1 = std::pow(x1,2)*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

            u2 = std::pow(x2,2)*std::pow((1 - x2),2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

            du11 = 7*std::pow(x1,2)*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + std::pow(x1,2)*(2*x1 - 2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha + 2*x1*std::pow((1 - x1),2)*(std::pow(x2,2)*(2*x2 - 2) + 2*x2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

            du12 = std::pow(x1,2)*std::pow((1 - x1),2)*(2*std::pow(x2,2) + 4*x2*(2*x2 - 2) + 2*std::pow((1 - x2),2))*std::exp(7*x1)/alpha;

            du21 = std::pow(x2,2)*std::pow((1 - x2),2)*(-49*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - 14*std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow(x1,2)*std::exp(7*x1) - 28*x1*std::pow((1 - x1),2)*std::exp(7*x1) - 4*x1*(2*x1 - 2)*std::exp(7*x1) - 2*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

            du22 = std::pow(x2,2)*(2*x2 - 2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha + 2*x2*std::pow((1 - x2),2)*(-7*std::pow(x1,2)*std::pow((1 - x1),2)*std::exp(7*x1) - std::pow(x1,2)*(2*x1 - 2)*std::exp(7*x1) - 2*x1*std::pow((1 - x1),2)*std::exp(7*x1))/alpha;

            pressure = 0.0;

            const double press_grad1 = 0.0;
            const double press_grad2 = 0.0;

            pressure_on_gauss_points[g] = pressure;
            velocity_on_gauss_points(0,g) = u1;
            velocity_on_gauss_points(1,g) = u2;
            pressure_gradient_on_gauss_points(0,g) = press_grad1;
            pressure_gradient_on_gauss_points(1,g) = press_grad2;
            velocity_gradient_on_gauss_points[g](0,0) = du11;
            velocity_gradient_on_gauss_points[g](0,1) = du21;
            velocity_gradient_on_gauss_points[g](1,0) = du12;
            velocity_gradient_on_gauss_points[g](1,1) = du22;
        }

        mExactScalar[id_elem-1] = pressure_on_gauss_points;
        mExactVector[id_elem-1] = velocity_on_gauss_points;
        mExactScalarGradient[id_elem-1] = pressure_gradient_on_gauss_points;
        mExactVectorGradient[id_elem-1] = velocity_gradient_on_gauss_points;

    }
}

/* Private functions ****************************************************/

};  // namespace Kratos.