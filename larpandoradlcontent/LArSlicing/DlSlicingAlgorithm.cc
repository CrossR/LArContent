/**
 *  @file   larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning slicing algorithm.
 *
 *  $Log: $
 */

#include <chrono>
#include <cmath>
#include <vector>

#include "Geometry/LArTPC.h"
#include "Objects/CartesianVector.h"
#include "Pandora/Pandora.h"

#include "Pandora/PandoraInternal.h"
#include "Pandora/StatusCodes.h"
#include "larpandoracontent/LArHelpers/LArFileHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

#include "larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.h"
#include "larpandoradlcontent/LArSlicing/HoughFinder.h"
#include "larpandoradlcontent/LArSlicing/KnnKDTree.h"

#include <Eigen/Dense>
#include <c10/core/TensorOptions.h>
#include <torch/script.h>
#include <torch/torch.h>

#define HEP_EVD_PANDORA_HELPERS 1
#include "hep_evd.h"

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlSlicingAlgorithm::DlSlicingAlgorithm() :
    m_scalingFactor{-1.0f},
    m_thresholds{},
    m_nDistanceClasses{-1},
    m_k{4}
{
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::Run()
{
    std::cout << "Starting DL Slicing Algorithm..." << std::endl;
    return this->Infer();
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::Infer()
{
    const CaloHitList *pCaloHitList{nullptr};
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListName, pCaloHitList));

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<CartesianVector> nodes;
    std::vector<std::array<float, 1>> node_features;
    std::vector<std::pair<int, int>> edges;
    this->GetGraphData(*pCaloHitList, nodes, node_features, edges);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Getting graph data took " << duration << " ms." << std::endl;

    LArDLHelper::TorchInputVector inputs;
    t1 = std::chrono::high_resolution_clock::now();
    this->BuildGraph(inputs, nodes, node_features, edges);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Building graph took " << duration << " ms." << std::endl;

    LArDLHelper::TorchMultiOutput output;
    t1 = std::chrono::high_resolution_clock::now();
    LArDLHelper::Forward(m_modelFile, inputs, output);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Inference took " << duration << " ms." << std::endl;

    // Now, we can process the outputs.
    // We should have 3 things:
    //
    // 1) Semantic distance labels for each node (i.e. hit) which is a class up
    //  to m_numSemanticClasses. This represents the distance of each hit from
    //  its primary neutrino vertex.
    //
    // 2) The raw embeddings for each node. We don't need to do anything with
    // these, except use them later on.
    //
    // 3) The position embeddings, same as above. Just saves re-computing them.
    const auto &outputTuple = output.toTuple();
    const auto &semanticLabels{outputTuple->elements()[0].toTensor()};
    const auto &rawEmbeddings{outputTuple->elements()[1].toTensor()};
    const auto &posEmbeddings{outputTuple->elements()[2].toTensor()};

    // Lets do some basic checks...
    std::cout << "Semantic Labels: " << semanticLabels.sizes() << ", " << semanticLabels.dtype() << std::endl;
    std::cout << "Raw Embeddings: " << rawEmbeddings.sizes() << ", " << rawEmbeddings.dtype() << std::endl;
    std::cout << "Pos Embeddings: " << posEmbeddings.sizes() << ", " << posEmbeddings.dtype() << std::endl;

    // DEBUG: Add visualization of the semantic labels to EVD, to check they
    // look sensible before we try to do any more complicated processing.
    const auto argMaxLabels = torch::argmax(semanticLabels, 1);
    HepEVD::Hits hitsToVis;

    int hitIdx{0};
    for (const auto pCaloHit : *pCaloHitList)
    {
        if (nullptr == pCaloHit)
            continue;

        const double label = argMaxLabels[hitIdx].item<double>();

        const auto x = pCaloHit->GetPositionVector().GetX();
        const auto y = pCaloHit->GetPositionVector().GetY();
        const auto z = pCaloHit->GetPositionVector().GetZ();
        const auto e = pCaloHit->GetInputEnergy();

        HepEVD::Hit *evdHit = new HepEVD::Hit({x, y, z}, e);
        evdHit->addProperties({{"SemanticLabel", label}});

        if (label <= 2)
            evdHit->addProperties({{"SeedCandidate", 1}});

        hitsToVis.push_back(evdHit);

        hitIdx++;
    }

    HepEVD::getServer()->addHits(hitsToVis);

    // Next, process the semantic labels with the Hough Transform to find vertex
    // candidates.
    t1 = std::chrono::high_resolution_clock::now();
    const unsigned int numHits = semanticLabels.size(0);
    const auto contiguousSemanticLabels = semanticLabels.contiguous();
    std::vector<float> semanticLabelsVec(
        contiguousSemanticLabels.data_ptr<float>(), contiguousSemanticLabels.data_ptr<float>() + (numHits * m_nDistanceClasses));

    // Setup and run the Hough Transform vertex finder.
    FastHoughFinder houghFinder(m_thresholds, m_scalingFactor);
    std::vector<CartesianVector> foundVertices = houghFinder.Fit(nodes, semanticLabelsVec);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Hough Transform vertex finding took " << duration << " ms." << std::endl;

    // DEBUG: Add them to HepEVD.
    HepEVD::Markers pointsToVis;
    for (const auto &vertex : foundVertices)
    {
        HepEVD::Point *evdPoint = new HepEVD::Point({vertex.GetX(), vertex.GetY(), vertex.GetZ()});
        pointsToVis.push_back(*evdPoint);
    }

    HepEVD::getServer()->addMarkers(pointsToVis);
    HepEVD::saveState("FoundVertices");
    HepEVD::startServer();

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

void GetVolumeProps(const Eigen::Vector4f &bounds, float &centerX, float &centerZ, float &widthX, float &widthZ)
{
    centerX = (bounds(0) + bounds(1)) / 2.0f;
    widthX = bounds(1) - bounds(0);
    centerZ = (bounds(2) + bounds(3)) / 2.0f;
    widthZ = bounds(3) - bounds(2);
}

//-----------------------------------------------------------------------------------------------------------------------------------------

void DlSlicingAlgorithm::BuildVolumeStitchingEdges(const std::vector<CartesianVector> &pos, std::vector<std::pair<int, int>> &edges)
{
    const auto &geoManager = this->GetPandora().GetGeometry();
    std::vector<const pandora::LArTPC *> tpcs;
    tpcs.reserve(geoManager->GetLArTPCMap().size());
    for (const auto &mapEntry : geoManager->GetLArTPCMap())
    {
        tpcs.push_back(mapEntry.second);
    }

    std::cout << "Number of Pandora TPC volumes: " << tpcs.size() << std::endl;

    // Find true max gap pairwise
    float max_gap = 0.0f;
    for (size_t i = 0; i < tpcs.size(); ++i)
    {
        for (size_t j = i + 1; j < tpcs.size(); ++j)
        {
            float gap_x =
                std::max(0.0f, std::abs(tpcs[i]->GetCenterX() - tpcs[j]->GetCenterX()) - (tpcs[i]->GetWidthX() + tpcs[j]->GetWidthX()) / 2.0f);
            float gap_z =
                std::max(0.0f, std::abs(tpcs[i]->GetCenterZ() - tpcs[j]->GetCenterZ()) - (tpcs[i]->GetWidthZ() + tpcs[j]->GetWidthZ()) / 2.0f);

            // TODO: The 15.0 here is just to avoid picking up the gaps between
            // say module 1 and 3...which is where module 2 is. Revist once
            // there is a fixed / better LArTPC input to use.
            if (gap_x > 0.0f && gap_x < 15.0f && gap_x > max_gap)
                max_gap = gap_x;
            if (gap_z > 0.0f && gap_z < 15.0f && gap_z > max_gap)
                max_gap = gap_z;
        }
    }

    if (max_gap <= 0.0f)
    {
        std::cout << "DLSlicingAlgorithm::BuildVolumeStitchingEdges - no gaps found between volumes, not adding stitching edges" << std::endl;
        return;
    }

    // Find hits near the gaps, such that we can stitch them.
    const float margin = 5.0f;
    std::vector<int> gap_indices;
    gap_indices.reserve(pos.size() / 10);

    for (size_t i = 0; i < pos.size(); ++i)
    {
        bool is_at_edge = false;
        for (const auto tpc : tpcs)
        {
            const float xMin = tpc->GetCenterX() - tpc->GetWidthX() / 2.0f;
            const float xMax = tpc->GetCenterX() + tpc->GetWidthX() / 2.0f;
            const float zMin = tpc->GetCenterZ() - tpc->GetWidthZ() / 2.0f;
            const float zMax = tpc->GetCenterZ() + tpc->GetWidthZ() / 2.0f;

            // Check if the hit is within 'margin' of the actual walls
            bool near_x_wall = (std::abs(pos[i].GetX() - xMin) <= margin) || (std::abs(pos[i].GetX() - xMax) <= margin);
            bool near_z_wall = (std::abs(pos[i].GetZ() - zMin) <= margin) || (std::abs(pos[i].GetZ() - zMax) <= margin);

            if (near_x_wall || near_z_wall)
            {
                is_at_edge = true;
                break;
            }
        }
        if (is_at_edge)
            gap_indices.push_back(i);
    }

    std::cout << "Found " << gap_indices.size() << " hits near volume edges." << std::endl;

    // Targeted Stitching Edges
    std::map<const pandora::LArTPC *, std::vector<int>> volumeToPointsMap;
    for (int idx : gap_indices)
    {
        for (const auto tpc : tpcs)
        {
            if (std::abs(pos[idx].GetX() - tpc->GetCenterX()) <= (tpc->GetWidthX() / 2.0f) + 0.1f &&
                std::abs(pos[idx].GetY() - tpc->GetCenterY()) <= (tpc->GetWidthY() / 2.0f) + 0.1f &&
                std::abs(pos[idx].GetZ() - tpc->GetCenterZ()) <= (tpc->GetWidthZ() / 2.0f) + 0.1f)
            {
                volumeToPointsMap[tpc].push_back(idx);
                break;
            }
        }
    }

    const float max_dist = max_gap * 1.5f;
    const float max_dist_sq = max_dist * max_dist;

    for (size_t i = 0; i < tpcs.size(); ++i)
    {
        for (size_t j = i + 1; j < tpcs.size(); ++j)
        {
            const auto tpcA = tpcs[i];
            const auto tpcB = tpcs[j];

            float gap_x = std::max(0.0f, std::abs(tpcA->GetCenterX() - tpcB->GetCenterX()) - (tpcA->GetWidthX() + tpcB->GetWidthX()) / 2.0f);
            float gap_z = std::max(0.0f, std::abs(tpcA->GetCenterZ() - tpcB->GetCenterZ()) - (tpcA->GetWidthZ() + tpcB->GetWidthZ()) / 2.0f);

            if (!((gap_x > 0 && gap_x < max_dist) || (gap_z > 0 && gap_z < max_dist)))
                continue;

            const auto &ptsA = volumeToPointsMap[tpcA];
            const auto &ptsB = volumeToPointsMap[tpcB];

            for (int idxA : ptsA)
            {
                float min_dist_A_to_B = std::numeric_limits<float>::max();
                int best_idx_B = -1;

                for (int idxB : ptsB)
                {
                    const float dist_sq = (pos[idxA] - pos[idxB]).GetMagnitudeSquared();
                    if (dist_sq < min_dist_A_to_B)
                    {
                        min_dist_A_to_B = dist_sq;
                        best_idx_B = idxB;
                    }
                }

                if (best_idx_B != -1 && min_dist_A_to_B < max_dist_sq)
                {
                    float min_dist_B_to_A = std::numeric_limits<float>::max();
                    int reciprocal_idx_A = -1;

                    for (int check_idxA : ptsA)
                    {
                        const float dist_sq = (pos[best_idx_B] - pos[check_idxA]).GetMagnitudeSquared();
                        if (dist_sq < min_dist_B_to_A)
                        {
                            min_dist_B_to_A = dist_sq;
                            reciprocal_idx_A = check_idxA;
                        }
                    }

                    if (idxA == reciprocal_idx_A)
                    {
                        edges.push_back({idxA, best_idx_B});
                        edges.push_back({best_idx_B, idxA});
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::GetGraphData(const CaloHitList &caloHits, std::vector<CartesianVector> &pos,
    std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges)
{
    // Build a KD-Tree for the hits, so we can do fast neighbour lookups when
    // building the graph.
    std::vector<KnnKdTree::KnnNode> knnNodes;

    // Reserve the space we know we need...
    knnNodes.reserve(caloHits.size());
    pos.reserve(caloHits.size());
    node_features.reserve(caloHits.size());
    edges.reserve(caloHits.size() * m_k);

    // Populate the KD-Tree nodes and node features from the CaloHits.
    int hitIdx{0};
    for (const auto pCaloHit : caloHits)
    {
        if (nullptr == pCaloHit)
            continue;

        const CartesianVector &hitPos = pCaloHit->GetPositionVector();

        knnNodes.push_back({{hitPos.GetX(), hitPos.GetY(), hitPos.GetZ()}, hitIdx++});
        pos.push_back(hitPos);
        node_features.push_back({pCaloHit->GetInputEnergy()});
    }

    // Build and use the KD-Tree to find nearest neighbours for edge construction.
    KnnKdTree kdTree(knnNodes);

    hitIdx = 0;
    for (const auto &node : knnNodes)
    {
        std::vector<int> neighbourIdxs = kdTree.FindNearestNeighbours(node, m_k);
        int edgeIdx = 0;
        for (const int neighbourIdx : neighbourIdxs)
            edges.push_back({node.original_id, neighbourIdx}); // Add reverse edge for undirected graph
    }

    std::cout << "Constructed graph with " << pos.size() << " nodes and " << edges.size() << " edges." << std::endl;

    // Next, make sure we add edges between hits on either side of TPC gaps, so the model can learn to stitch across them.
    this->BuildVolumeStitchingEdges(pos, edges);

    // Finally, remove any duplicate edges...
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::cout << "After coalescing edges, we now have " << pos.size() << " nodes and " << edges.size() << " edges." << std::endl;

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::BuildGraph(LArDLHelper::TorchInputVector &inputs, std::vector<CartesianVector> &pos,
    std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges)
{
    const int numNodes{static_cast<int>(pos.size())};
    const int numEdges{static_cast<int>(edges.size())};

    const int numFeatures{static_cast<int>(node_features[0].size())};
    const int edgeShape{4};

    const auto asFloat = torch::TensorOptions().dtype(torch::kFloat32);
    const auto asInt = torch::TensorOptions().dtype(torch::kInt64);

    LArDLHelper::TorchInput posTensor, xTensor, edgeIndexTensor, edgeAttrTensor;
    LArDLHelper::InitialiseInput({numNodes, 3}, posTensor, asFloat);
    LArDLHelper::InitialiseInput({2, numEdges}, edgeIndexTensor, asInt);
    LArDLHelper::InitialiseInput({numNodes, numFeatures}, xTensor, asFloat);
    LArDLHelper::InitialiseInput({numEdges, edgeShape}, edgeAttrTensor, asFloat);

    // Also create a batch tensor.
    // In python/training land...this is a tensor that tells the model how many graphs are in the batch, and which
    // nodes/edges belong to which graph.
    // In this case, we only have one graph, so we can just set it to 0 for all nodes and edges.
    torch::Tensor batchTensor = torch::zeros(numNodes, torch::kLong);

    // Use raw memory pointers to access the various tensors, to massively speed
    // up writing.
    float *posTensorPtr = posTensor.data_ptr<float>();
    float *xTensorPtr = xTensor.data_ptr<float>();
    int64_t *edgeIndexTensorPtr = edgeIndexTensor.data_ptr<int64_t>();
    float *edgeAttrTensorPtr = edgeAttrTensor.data_ptr<float>();

    // First, the nodes...
    for (int i = 0; i < numNodes; ++i)
    {
        posTensorPtr[i * 3 + 0] = pos[i].GetX();
        posTensorPtr[i * 3 + 1] = pos[i].GetY();
        posTensorPtr[i * 3 + 2] = pos[i].GetZ();
        xTensorPtr[i] = node_features[i][0];
    }

    // Then, the edges...
    for (int i = 0; i < numEdges; ++i)
    {
        // First, we just set the edge indices.
        edgeIndexTensorPtr[i] = edges[i].first;
        edgeIndexTensorPtr[numEdges + i] = edges[i].second;

        // We also need to calculate the edge attributes...
        const auto &posA{pos[edges[i].first]};
        const auto &posB{pos[edges[i].second]};

        const auto relativePos = posA - posB;

        // Apply scaling factor to the relative position and distance.
        // This ensures the features match the trained model's expected input
        // scale, which can improve performance and stability.
        const float scaledRelX = relativePos.GetX() / m_scalingFactor;
        const float scaledRelY = relativePos.GetY() / m_scalingFactor;
        const float scaledRelZ = relativePos.GetZ() / m_scalingFactor;
        const float scaledDist = std::sqrt(scaledRelX * scaledRelX + scaledRelY * scaledRelY + scaledRelZ * scaledRelZ);

        // Store the edge attributes...
        edgeAttrTensorPtr[i * edgeShape + 0] = scaledRelX;
        edgeAttrTensorPtr[i * edgeShape + 1] = scaledRelY;
        edgeAttrTensorPtr[i * edgeShape + 2] = scaledRelZ;
        edgeAttrTensorPtr[i * edgeShape + 3] = scaledDist;
    }

    // Finally, stick them all together into the input vector.
    inputs.insert(inputs.end(), {xTensor, posTensor, edgeIndexTensor.contiguous(), edgeAttrTensor, batchTensor});

    // Print some debug information
    std::cout << "DlSlicingAlgorithm::BuildGraph: Built graph with " << numNodes << " nodes, " << numEdges << " edges, and " << numFeatures
              << " features per node." << std::endl;
    std::cout << "Nodes: " << posTensor.sizes() << ", " << posTensor.dtype() << std::endl;
    std::cout << "Features: " << xTensor.sizes() << ", " << xTensor.dtype() << std::endl;
    std::cout << "Edges: " << edgeIndexTensor.sizes() << ", " << edgeIndexTensor.dtype() << std::endl;
    std::cout << "Edge Features: " << edgeShape << std::endl;

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

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "DistanceThresholds", m_thresholds));

    m_nDistanceClasses = m_thresholds.size() + 1; // We have one more class than thresholds, as the thresholds define the boundaries between classes.

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_dl_content
