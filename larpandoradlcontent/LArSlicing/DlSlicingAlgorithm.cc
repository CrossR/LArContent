/**
 *  @file   larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning slicing algorithm.
 *
 *  $Log: $
 */

#include <c10/core/TensorOptions.h>
#include <chrono>
#include <cmath>

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include "Objects/CartesianVector.h"
#include "Pandora/Pandora.h"

#include "Pandora/PandoraInternal.h"
#include "Pandora/StatusCodes.h"
#include "larpandoracontent/LArHelpers/LArFileHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArObjects/LArGraph.h"
#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

#include "larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.h"

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlSlicingAlgorithm::DlSlicingAlgorithm() :
    m_modelFile{""},
    m_scalingFactor{-1.0f}
{
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::Run()
{
    return this->Infer();
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::Infer()
{
    const CaloHitList *pCaloHitList{nullptr};
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListName, pCaloHitList));

    std::vector<CartesianVector> nodes;
    std::vector<std::array<float, 1>> node_features;
    std::vector<std::pair<int, int>> edges;
    this->GetGraphData(*pCaloHitList, nodes, node_features, edges);

    LArDLHelper::TorchInputVector inputs;
    this->BuildGraph(inputs, nodes, node_features, edges);

    LArDLHelper::TorchOutput output;
    LArDLHelper::Forward(m_modelFile, inputs, output);

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::GetGraphData(const CaloHitList &caloHits, std::vector<CartesianVector> &pos,
    std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges)
{

    // Build a N, 3 matrix of hit positions...
    Eigen::MatrixXf hitMatrix(caloHits.size(), 3);
    unsigned int i{0};

    for (const auto pCaloHit : caloHits)
    {
        if (nullptr == pCaloHit)
            continue;

        const CartesianVector &hitPos{pCaloHit->GetPositionVector()};
        hitMatrix.col(i) << hitPos.GetX(), hitPos.GetY(), hitPos.GetZ();
        ++i;
    }

    // We can now use the hit matrix to create the input for the model.
    const unsigned int k(4);

    // For every hit, find its k nearest neighbors.
    // With that, we can add the node to the graph with:
    // - Its position into the pos vector.
    // - Its features into the x vector
    // - The edges to the k nearest neighbors into the edges vector.
    for (unsigned int r = 0; r < hitMatrix.rows(); ++r)
    {
        Eigen::RowVectorXf row(3);
        row << hitMatrix(r, 0), hitMatrix(r, 1), hitMatrix(r, 2);

        // Calculate the distance to all other hits
        Eigen::MatrixXf diffs((hitMatrix.rowwise() - row));
        Eigen::VectorXf dists_sq(diffs.rowwise().squaredNorm());

        // Set the distance to itself to max
        dists_sq(r) = std::numeric_limits<float>::max();

        std::vector<int> knnIndices;
        std::iota(knnIndices.begin(), knnIndices.end(), 0);
        std::partial_sort(knnIndices.begin(), knnIndices.begin() + k, knnIndices.end(),
            [&dists_sq](int i1, int i2) { return dists_sq[i1] < dists_sq[i2]; });
        knnIndices.resize(k);

        //  Now, we can insert the edges into the edges vector.
        for (const int index : knnIndices)
            edges.emplace_back(r, index);

        // We've now found the k nearest neighbors for this hit.
        // Finally, just update the pos and x vectors.
        const auto &pCaloHit{*std::next(caloHits.begin(), r)};
        const auto &posVector{pCaloHit->GetPositionVector()};
        pos.emplace_back(posVector);
        node_features.emplace_back(std::array<float, 1>{pCaloHit->GetMipEquivalentEnergy()});
    }

    // Now, we have a full graph with a KNN structure.
    // We just need to update it to also include so-called "stitching edges".
    // These are edges that connect over the gaps between detector modules.
    // To do that, we just get the detector geometry, and find all the hits that
    // within some distance of the gaps. We can then do a simple distance check
    // to see if the hits are close enough to be connected by an edge.
    const auto &geoManager{this->GetPandora().GetGeometry()};
    Eigen::MatrixXf tpcBounds(4, geoManager->GetLArTPCMap().size());

    i = 0;
    for (const auto &tpc : geoManager->GetLArTPCMap())
    {
        const float xCenter{tpc.second->GetCenterX()};
        const float xHalfWidth{tpc.second->GetWidthX() / 2.0f};

        const float zCenter{tpc.second->GetCenterZ()};
        const float zHalfWidth{tpc.second->GetWidthZ() / 2.0f};

        tpcBounds.col(i) << xCenter - xHalfWidth, xCenter + xHalfWidth, zCenter - zHalfWidth, zCenter + zHalfWidth;
        ++i;
    }

    // Now, we can iterate over the hits and check if they are within the bounds of any TPC.
    std::vector<int> gapHitIndices;
    const float margin(5.0f); // Margin to consider a hit as being in the gap
    i = 0;
    for (const auto &pCaloHit : caloHits)
    {
        const auto &posVector{pCaloHit->GetPositionVector()};
        const float x{posVector.GetX()};
        const float z{posVector.GetZ()};

        bool isAtEdge{false};
        for (const auto &bounds : tpcBounds.colwise())
        {
            const bool isNearX{bounds(0) - margin <= x && x <= bounds(1) + margin};
            const bool isNearZ{bounds(2) - margin <= z && z <= bounds(3) + margin};

            if (isNearX || isNearZ)
            {
                isAtEdge = true;
                break;
            }
        }

        if (isAtEdge)
            gapHitIndices.push_back(i);

        ++i;
    }

    // Now, we essentially repeat the same thing as before, but instead of a KNN, we just
    // do a simple distance check to see if the hits are close enough to be connected by an edge.
    Eigen::MatrixXf gapHitMatrix(gapHitIndices.size(), 3);

    for (const int index : gapHitIndices)
    {
        if (index < 0 || index >= static_cast<int>(caloHits.size()))
        {
            std::cout << "DlSlicingAlgorithm::MakeNetworkInputFromHits: Index out of bounds for calo hits: " << index << std::endl;
            return STATUS_CODE_INVALID_PARAMETER;
        }

        const auto &pCaloHit{*std::next(caloHits.begin(), index)};

        const auto &posVector{pCaloHit->GetPositionVector()};
        gapHitMatrix.col(index) << posVector.GetX(), posVector.GetY(), posVector.GetZ();
    }

    // Iterate over all the gaps hits, and find all the hits that are within a certain distance of them.
    for (const int index : gapHitIndices)
    {
        const auto &pCaloHit{*std::next(caloHits.begin(), index)};
        const auto &posVector{pCaloHit->GetPositionVector()};
        Eigen::RowVectorXf row(3);
        row << posVector.GetX(), posVector.GetY(), posVector.GetZ();

        Eigen::MatrixXf diffs((gapHitMatrix.rowwise() - row));
        Eigen::VectorXf dists_sq(diffs.rowwise().squaredNorm());

        // Set the distance to itself to max
        dists_sq(index) = std::numeric_limits<float>::max();

        // Get every hit that is within a certain distance of the gap hit.
        const float maxDistance(10.0f);
        for (unsigned int j = 0; j < gapHitMatrix.rows(); ++j)
        {
            if (gapHitMatrix[index] == gapHitMatrix[j])
                continue;

            if (dists_sq(j) < maxDistance * maxDistance)
                edges.emplace_back(index, j);
        }
    }

    // Almost there. We want to coalesce the edges down to remove any duplicates.
    std::set<std::pair<int, int>> uniqueEdges;
    for (const auto &edge : edges)
    {
        if (edge.first == edge.second)
            continue;

        if (edge.first < edge.second)
            uniqueEdges.emplace(edge.first, edge.second);
        else
            uniqueEdges.emplace(edge.second, edge.first);
    }

    edges.clear();
    for (const auto &edge : uniqueEdges)
    {
        edges.emplace_back(edge.first, edge.second);
        edges.emplace_back(edge.second, edge.first);
    }

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::BuildGraph(LArDLHelper::TorchInputVector &inputs, std::vector<CartesianVector> &pos,
    std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges)
{
    const int numNodes{static_cast<int>(pos.size())};
    const int numEdges{static_cast<int>(edges.size())};

    const int numFeatures{static_cast<int>(node_features[0].size())};
    const int edgeShape{2};

    const auto asFloat = torch::TensorOptions().dtype(torch::kFloat32);
    const auto asInt = torch::TensorOptions().dtype(torch::kInt64);

    LArDLHelper::TorchInput posTensor, xTensor, edgeIndexTensor;
    LArDLHelper::InitialiseInput({numNodes, 3}, posTensor, asFloat);
    LArDLHelper::InitialiseInput({edgeShape, numEdges}, edgeIndexTensor, asInt);
    LArDLHelper::InitialiseInput({numNodes, numFeatures}, xTensor, asFloat);

    // First, the nodes...
    for (int i = 0; i < numNodes; ++i)
    {
        const auto &nodePos{pos[i]};

        posTensor.slice(0, i, i + 1) = torch::tensor({nodePos.GetX(), nodePos.GetY(), nodePos.GetZ()}, asFloat);
        xTensor.slice(0, i, i + 1) = torch::tensor({node_features[i]}, asFloat);
    }

    // Then, the edges...
    for (int i = 0; i < numEdges; ++i)
    {
        edgeIndexTensor[0][i] = edges[i].first;
        edgeIndexTensor[1][i] = edges[i].second;
    }

    // Finally, stick them all together into the input vector.
    inputs.insert(inputs.end(), {posTensor, xTensor, edgeIndexTensor});

    // Print some debug information
    std::cout << "DlSlicingAlgorithm::BuildGraph: Built graph with " <<
        numNodes << " nodes, " << numEdges << " edges, and " << numFeatures << " features per node." << std::endl;
    std::cout << "Nodes: " << posTensor.sizes() << ", " << posTensor.dtype() << std::endl;
    std::cout << "Features: " << xTensor.sizes() << ", " << xTensor.dtype() << std::endl;
    std::cout << "Edges: " << edgeIndexTensor.sizes() << ", " << edgeIndexTensor.dtype() << std::endl;

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    std::string modelName;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileName", modelName));
    modelName = LArFileHelper::FindFileInPath(modelName, "FW_SEARCH_PATH");
    LArDLHelper::LoadModel(modelName, m_modelFile);

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ScalingFactor", m_scalingFactor));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputCaloHitListName", m_caloHitListName));

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_dl_content
