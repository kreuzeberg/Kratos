# function2
from re import X
import sys
import time
import importlib
import numpy as np
import math
# from gid_output_process import GiDOutputProcess

import KratosMultiphysics
from KratosMultiphysics import *


def Create_Fluid_model_part(divisions):
    current_model = KratosMultiphysics.Model()
    model_part = current_model.CreateModelPart("ModelPart")
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISTANCE)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.NODAL_AREA)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY_X)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY_X_GRADIENT)
    model_part.ProcessInfo.SetValue(KratosMultiphysics.DOMAIN_SIZE, 2)

    problem_domain = KratosMultiphysics.Quadrilateral2D4(
        KratosMultiphysics.Node(1, -4.25, -4.18, 0.0),
        KratosMultiphysics.Node(2, -4.25,  4.0, 0.0),
        KratosMultiphysics.Node(3,  7.0,  4.0, 0.0),
        KratosMultiphysics.Node(4,  7.0, -4.18, 0.0))
    parameters = KratosMultiphysics.Parameters("{}")
    parameters.AddEmptyValue("element_name").SetString("Element2D3N")
    # parameters.AddEmptyValue("condition_name").SetString("LineCondition2D2N")
    parameters.AddEmptyValue("create_skin_sub_model_part").SetBool(False)
    parameters.AddEmptyValue("number_of_divisions").SetInt(divisions)

    KratosMultiphysics.StructuredMeshGeneratorProcess(problem_domain, model_part, parameters).Execute()
    return model_part

def Import_Background_model_part(Model,name_background_mpda):
    # Set skin_model_part geometry
    # current_model = KratosMultiphysics.Model()
    model_part = Model.CreateModelPart("fluid_model_part")
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISTANCE)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.NODAL_AREA)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.TEMPERATURE)
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.TEMPERATURE_GRADIENT)
    model_part.ProcessInfo.SetValue(KratosMultiphysics.DOMAIN_SIZE, 2)
    KratosMultiphysics.ModelPartIO(name_background_mpda).ReadModelPart(model_part)
    return model_part

def Import_Structural_model_part(name_structural_mpda):
    # Set skin_model_part geometry
    current_model = KratosMultiphysics.Model()
    skin_model_part = current_model.CreateModelPart("skin_model_part")
    skin_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.TEMPERATURE)
    skin_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY_X)
    skin_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY_Y)
    KratosMultiphysics.ModelPartIO(name_structural_mpda).ReadModelPart(skin_model_part)
    return skin_model_part

# def Import_Structural_SUB_model_part(model_part, name_structural_mpda):
#     # Set skin_model_part geometry
#     skin_model_part = model_part.CreateSubModelPart("skin_model_part")
#     skin_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.TEMPERATURE)
#     KratosMultiphysics.ModelPartIO(name_structural_mpda).ReadModelPart(skin_model_part)
#     return skin_model_part, model_part

def Find_surrogate_nodes(model_part,iter):
    start_time = time.time()
    # General methods for find the surrogate nodes
    name_surrogate_sub_model_part = "surrogate_sub_model_part" + "_" + str(iter)
    surrogate_sub_model_part = model_part.CreateSubModelPart(name_surrogate_sub_model_part)
    # second_level_surrogate_sub_model_part = model_part.CreateSubModelPart("second_level_surrogate_sub_model_part")
    count2 = 0
    # # Inside-Outside problem
    # for node in model_part.Nodes:
    #     a = - node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
    #     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE, a)
    for elem in model_part.Elements :
        count_pos = 0
        count_neg = 0
        for node in elem.GetGeometry() :
            phi = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            if phi > 0 :
                count_pos = count_pos + 1
            else :
                count_neg = count_neg + 1
        # When count_neg*count_pos != 0 --> the element is cut
        if count_neg * count_pos != 0 :
            for node in elem.GetGeometry() :
                if node.GetSolutionStepValue(KratosMultiphysics.DISTANCE) > 0 :
                    node.Set(VISITED, True)
        if count_pos == 3 :
            # The element is a fluid element completely ouside the surrogate boundary
            elem.Set(MARKER,True)
            surrogate_sub_model_part.AddElement(elem,0)
            for node in elem.GetGeometry() :
                node.Set(MARKER, True)
            count2 = count2 + 1
        # if count_neg > 0 :
        #     model_part.RemoveElement(elem)
    print('Total number of "Fluid" element : ', count2)
    # Count the number of surrogate nodes
    tot_sur_nodes = 0
    for node in model_part.Nodes :
        if node.Is(VISITED):
            surrogate_sub_model_part.AddNode(node,0)
            tot_sur_nodes = tot_sur_nodes +1 
    print('Number of surrogate nodes: ',tot_sur_nodes)
    # Discretize if an element is "boundary" so if it has at least one node that is surrogate
    for elem in surrogate_sub_model_part.Elements :
        for node in elem.GetGeometry() :
            if node.Is(VISITED) :
                elem.Set(BOUNDARY, True)
                break
    print("--> %s seconds for Find_surrogate_nodes" % (time.time() - start_time))
    return surrogate_sub_model_part,tot_sur_nodes



def Find_closest_skin_element(model_part,skin_model_part,tot_sur_nodes,tot_skin_el) :
    start_time = time.time()
    # if we are interested in the closest skin ELEMENT
    # file_due = open("closest_skin_element.txt", "w")
    # Inizializzo array closest_element
    closest_element = [0] * tot_sur_nodes   # the number of element and nodes of 
                                            # the skin is the same in this case
    i = 0
    for node in model_part.Nodes :
        if node.Is(VISITED):
            surr_node = node.Id
            # Inizializzo array equation_el
            equation_el = [0] * tot_skin_el
            for j in range(tot_skin_el) :  # Run over the skin ELEMENTS
                node1 = skin_model_part.Conditions[j+1].GetNodes()[0]
                node2 = skin_model_part.Conditions[j+1].GetNodes()[1]
                equation_el[j] = (node1.X-model_part.Nodes[surr_node].X)**2  + \
                                (node1.Y-model_part.Nodes[surr_node].Y)**2 + \
                                (node2.X-model_part.Nodes[surr_node].X)**2 + \
                                (node2.Y-model_part.Nodes[surr_node].Y)**2
            index_min = np.argmin(equation_el) # j-esimo nodo
            closest_element[i] = skin_model_part.Conditions[index_min+1].Id
            # file_due.write(str(closest_element[i]))
            # file_due.write('\n')
            i = i + 1
    # file_due.close()
    print("--> %s seconds for Find_closest_skin_element" % (time.time() - start_time))
    return closest_element



def Find_projections(model_part,skin_model_part,tot_sur_nodes,tot_skin_el,closest_element) :
    start_time = time.time()
    # Find the PROJECTION for each of the surrogate nodes on the closest skin element
    # 1.0.1 Initialize a matrix with the coordinates of the projections
    projection_surr_nodes = [[0 for _ in range(2)] for _ in range(tot_sur_nodes)]
    # file_tre = open("projection_surr_nodes.txt", "w")
    # 1.1 Run over each surrogate node take the closest skin element
    i = 0
    for node in model_part.Nodes :
        if node.Is(VISITED) :
            # 1.2 take the closest skin element --> closest_element[i]
            # 1.3 Get the two nodes
            node1 = skin_model_part.Conditions[closest_element[i]].GetNodes()[0]
            node2 = skin_model_part.Conditions[closest_element[i]].GetNodes()[1]
            # 1.4 Compute m and q 
            if (node1.X - node2.X)!=0 :
                m = (node1.Y - node2.Y) / (node1.X - node2.X)
                q = node1.Y - m * node1.X
                # 1.5 Compute Q
                if m != 0 :
                    Q = node.Y + 1/m * node.X
                    projection_surr_nodes[i][0] = (Q-q) / (m+1/m)
                else : 
                    # La retta perpendicolare è del tipo x = node.X
                    projection_surr_nodes[i][0] = node.X
                projection_surr_nodes[i][1] = m * projection_surr_nodes[i][0] + q
            else :
                # La retta per i nodi 1 e 2 è verticale -> Trovo subito le proiezioni
                projection_surr_nodes[i][0] = node1.X
                projection_surr_nodes[i][1] = node.Y
            # 1.7 Need to check if the point actually lies on the closest elements: compute the distance from
                # each skin node of the closest element and check that is less than the length of the element.
            check1 = (projection_surr_nodes[i][0] - node1.X)**2 + (projection_surr_nodes[i][1] - node1.Y)**2
            check2 = (projection_surr_nodes[i][0] - node2.X)**2 + (projection_surr_nodes[i][1] - node2.Y)**2
            element_length = (node1.X - node2.X)**2 + (node1.Y - node2.Y)**2
            if check1 > element_length or check2 > element_length :
                # print('-->\nNeed projection correction for node: ', node.Id)
                # Need to find the real projection
                # Take the closest node between node1 and node2
                if check1 < check2 :
                    candidate = node1
                else :
                    candidate = node2
                #-----------------------NEED TO IMPROVE--------------------------------------- 
                # search for the element with node = candidate (so I find the other unique possible second-closest element)
                for elem in skin_model_part.Conditions:
                    for nod in elem.GetGeometry():
                        if nod.Id == candidate.Id :
                            break
                    if nod.Id == candidate.Id :
                        break
                # Now we know on which element the projection lies --> We take the two nodes: node1 & node2
                node1 = skin_model_part.Conditions[elem.Id].GetNodes()[0]
                node2 = skin_model_part.Conditions[elem.Id].GetNodes()[1]
                # 1.4 Compute m and q 
                if (node1.X - node2.X)!=0 :
                    m = (node1.Y - node2.Y) / (node1.X - node2.X)
                    q = node1.Y - m * node1.X
                    # 1.5 Compute Q
                    if m != 0 :
                        Q = node.Y + 1/m * node.X
                        projection_surr_nodes[i][0] = (Q-q) / (m+1/m)
                    else : 
                        # La retta perpendicolare è del tipo x = node.X
                        projection_surr_nodes[i][0] = node.X
                    projection_surr_nodes[i][1] = m * projection_surr_nodes[i][0] + q
                else :
                    # La retta per i nodi 1 e 2 è verticale -> Trovo subito le proiezioni
                    projection_surr_nodes[i][0] = node1.X
                    projection_surr_nodes[i][1] = node.Y
                # SECOND CHECK (the projection, actually lies on an element?)
                check1 = (projection_surr_nodes[i][0] - node1.X)**2 + (projection_surr_nodes[i][1] - node1.Y)**2
                check2 = (projection_surr_nodes[i][0] - node2.X)**2 + (projection_surr_nodes[i][1] - node2.Y)**2
                element_length = (node1.X - node2.X)**2 + (node1.Y - node2.Y)**2
                if check1 > element_length or check2 > element_length :
                    print('Need a second projection correction for node: ', node.Id)
                    # No, the projection just found does not lie on a skin element
                    # --> Take the closest node as the projection
                    projection_surr_nodes[i][0] = candidate.X
                    projection_surr_nodes[i][1] = candidate.Y
                    node.Set(INTERFACE, True)
                else :
                    # We have found the correct projection
                    print('Trovata proiezione nel secondo elemento più vicino del nodo : ', node.Id)
            # file_tre.write(str(projection_surr_nodes[i][0]))
            # file_tre.write('  ')
            # file_tre.write(str(projection_surr_nodes[i][1]))
            # file_tre.write('\n')
            i = i + 1
    # file_tre.close()
    print("--> %s seconds for Find_projections" % (time.time() - start_time))
    return projection_surr_nodes



def Dirichlet_BC (model_part,skin_model_part,tot_sur_nodes,closest_element,projection_surr_nodes) :
    # Compute the value of the velocity_x on the surrogate boundary using the grandient and the distance
    surr_BC = [0] * tot_sur_nodes
    i = 0
    for node in model_part.Nodes :
        if node.Is(VISITED) :
            # Need the distance vector : x - x_tilde
            # 1.1 Initialize a Kratos vector
            d_vector = KratosMultiphysics.Array3()
            # 1.2 Compute the vector x - x_tilde
            d_vector[0] = projection_surr_nodes[i][0] - node.X
            d_vector[1] = projection_surr_nodes[i][1] - node.Y
            d_vector[2] = 0.0
            # 1.3 Take the gradient of the generic scalar vector field
            grad = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT)
            # 1.4 Compute the scalar product
            scalar_product = 0
            for j in range(2) :
                scalar_product = scalar_product +  d_vector[j] * grad[j]
            correction = scalar_product
            # 1.5 Obtain the value of the scalar field at the projection with an INTERPOLATION
            # for node in skin_model_part.Nodes :
            #     print(node.X, node.Y)
            node1 = skin_model_part.Conditions[closest_element[i]].GetNodes()[0]
            node2 = skin_model_part.Conditions[closest_element[i]].GetNodes()[1]
            d_star = math.sqrt((node1.X-projection_surr_nodes[i][0])**2 + (node1.Y-projection_surr_nodes[i][1])**2)
            d = math.sqrt((node1.X-node2.X)**2 + (node1.Y-node2.Y)**2)
            velocity = node1.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE) - d_star / d * (node1.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)-node2.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE))
            # 1.5.1 Additional check for the node with "second correction"
            if node.Is(INTERFACE) :
                if projection_surr_nodes[i][0] == node1.X and projection_surr_nodes[i][1] == node1.Y:
                    velocity = node1.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
                    # print('node 1')
                else :
                    # print(projection_surr_nodes[i][0], node2.X)
                    velocity = node2.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
                    # print('node 2')
            # 1.6 Compute the approximated Dirichlet BC at the surrogate boundary
            surr_BC[i] = velocity - correction
            # print('errore = ', abs(velocity - correction - node.X-node.Y))
            # print('\n')
            i = i+1
    return surr_BC


def Interpolation(skin_model_part,closest_element,projection_surr_nodes, i, node):
    node1 = skin_model_part.Conditions[closest_element[i]].GetNodes()[0]
    node2 = skin_model_part.Conditions[closest_element[i]].GetNodes()[1]
    d_star = math.sqrt((node1.X-projection_surr_nodes[i][0])**2 + (node1.Y-projection_surr_nodes[i][1])**2)
    d = math.sqrt((node1.X-node2.X)**2 + (node1.Y-node2.Y)**2)
    velocity = node1.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE) - d_star / d * (node1.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)-node2.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE))
    # 1.5.1 Additional check for the node with "second correction"
    if node.Is(INTERFACE) :
        if projection_surr_nodes[i][0] == node1.X and projection_surr_nodes[i][1] == node1.Y:
            velocity = node1.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
        else :
            velocity = node2.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
    return velocity



def Dirichlet_BC_CFD (model_part,skin_model_part,tot_sur_nodes,closest_element,projection_surr_nodes) :
    # Compute the value of the velocity_x on the surrogate boundary using the grandient and the distance
    surr_BC_x = [0] * tot_sur_nodes
    surr_BC_y = [0] * tot_sur_nodes
    i = 0
    for node in model_part.Nodes :
        if node.Is(VISITED) :
            # Need the distance vector : x - x_tilde
            # 1.1 Initialize a Kratos vector
            d_vector = KratosMultiphysics.Array3()
            # 1.2 Compute the vector x - x_tilde
            d_vector[0] = projection_surr_nodes[i][0] - node.X
            d_vector[1] = projection_surr_nodes[i][1] - node.Y
            d_vector[2] = 0.0
            # 1.3 Take the gradient of the generic scalar vector field
            grad_x = node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X_GRADIENT)
            grad_y = node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y_GRADIENT)
            # 1.4 Compute the scalar product
            scalar_product_x = 0
            scalar_product_y = 0
            for j in range(2) :
                scalar_product_x = scalar_product_x +  d_vector[j] * grad_x[j]
                scalar_product_y = scalar_product_y +  d_vector[j] * grad_y[j]
            correction_x = scalar_product_x
            correction_y = scalar_product_y
            # 1.5 Obtain the value of the scalar field at the projection with an INTERPOLATION
            node1 = skin_model_part.Conditions[closest_element[i]].GetNodes()[0]
            node2 = skin_model_part.Conditions[closest_element[i]].GetNodes()[1]
            d_star = math.sqrt((node1.X-projection_surr_nodes[i][0])**2 + (node1.Y-projection_surr_nodes[i][1])**2)
            d = math.sqrt((node1.X-node2.X)**2 + (node1.Y-node2.Y)**2)
            velocity_x = node1.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X) - d_star / d * (node1.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X)-node2.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X))
            velocity_y = node1.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y) - d_star / d * (node1.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y)-node2.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y))
            # 1.5.1 Additional check for the node with "second correction"
            if node.Is(INTERFACE) :
                if projection_surr_nodes[i][0] == node1.X and projection_surr_nodes[i][1] == node1.Y:
                    velocity_x = node1.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X)
                    velocity_y = node1.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y)
                    # print('node 1')
                else :
                    # print(projection_surr_nodes[i][0], node2.X)
                    velocity_x = node2.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X)
                    velocity_y = node2.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y)
                    # print('node 2')
            # 1.6 Compute the approximated Dirichlet BC at the surrogate boundary
            surr_BC_x[i] = velocity_x - correction_x
            surr_BC_y[i] = velocity_y - correction_y
            # print('errore = ', abs(velocity - correction - node.X-node.Y))
            # print('\n')
            i = i+1
    return surr_BC_x, surr_BC_y




def Create_sub_model_part_fluid(main_model_part,iter) :
    name_sub_model_part_fluid = "sub_model_part_fluid" + "_" + str(iter)
    sub_model_part_fluid = main_model_part.CreateSubModelPart(name_sub_model_part_fluid)
    for node in main_model_part.Nodes :
        if node.GetSolutionStepValue(KratosMultiphysics.DISTANCE) > 0 :
            sub_model_part_fluid.AddNode(node,0)
    for elem in main_model_part.Elements :
        if elem.Is(MARKER):
            sub_model_part_fluid.AddElement(elem,0)
    return sub_model_part_fluid



def Compute_error(main_model_part) :
        file_due = open("error.txt", "w")
        L2_err = 0
        H1_err_grad = 0
        KratosMultiphysics.ComputeNodalGradientProcess(
        main_model_part,
        KratosMultiphysics.TEMPERATURE,
        KratosMultiphysics.TEMPERATURE_GRADIENT,
        KratosMultiphysics.NODAL_AREA).Execute()
        total_number_fluid_nodes = 0
        for node in main_model_part.Nodes :
            exact = node.X + node.Y    # --> Tutto lineare
            # exact = 0.25*(4 - ((node.X)**2 + (node.Y)**2) )  # --> Paraboloide
            # exact = 0.25*(9-node.X**2-node.Y**2-2*math.log(3) + math.log(node.X**2+node.Y**2)) + 0.25 *math.sin(node.X) * math.sinh(node.Y) 
            # exact_grad = KratosMultiphysics.Array3()
            # exact_grad[0] = 0.25 * (-2*node.X + 2*node.X / (node.X**2 + node.Y**2)**2)  +  0.25 * math.cos(node.X) * math.sinh(node.Y)
            # exact_grad[1] = 0.25 * (-2*node.Y + 2*node.Y / (node.X**2 + node.Y**2)**2)  +  0.25 * math.sin(node.X) * math.cosh(node.Y)
            if node.IsNot(MARKER) :
                exact = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
                # exact_grad[0] = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT)[0]
                # exact_grad[1] = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT)[1]
            else :
                total_number_fluid_nodes = total_number_fluid_nodes + 1
            # L2_err = L2_err + (node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)-exact)**2
            # H1_err_grad = H1_err_grad + (node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)-exact)**2 + (node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT)[0]-exact_grad[0])**2 + (node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT)[1]-exact_grad[1])**2
            file_due.write(str(abs(node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)-exact)))
            file_due.write('\n')
        print('Total number of fluid nodes : ', total_number_fluid_nodes)
        if total_number_fluid_nodes == 0 :
            total_number_fluid_nodes = len(main_model_part.Nodes)
        L2_err = math.sqrt(L2_err) / total_number_fluid_nodes
        H1_err_grad = math.sqrt(H1_err_grad) / total_number_fluid_nodes
        file_due.write(str(L2_err))
        # print('Errore in norma L2 : ', L2_err)
        # print('Errore in norma H1 : ', H1_err_grad)
        file_due.close
        return