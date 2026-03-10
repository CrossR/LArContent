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
#include "larpandoracontent/LArUtility/KDTreeLinkerToolsT.h"
#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

#include "larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.h"

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
    m_nDistanceClasses{-1}
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

    // Finally, we can visualise the semanticLabels in HepEVD.
    const auto argMaxLabels = torch::argmax(semanticLabels, 1);
    HepEVD::Hits hitsToVis;

    for (const auto pCaloHit : *pCaloHitList)
    {
        const int hitIdx = std::distance(pCaloHitList->begin(), std::find(pCaloHitList->begin(), pCaloHitList->end(), pCaloHit));
        const double label = argMaxLabels[hitIdx].item<double>();

        const auto x = pCaloHit->GetPositionVector().GetX();
        const auto y = pCaloHit->GetPositionVector().GetY();
        const auto z = pCaloHit->GetPositionVector().GetZ();
        const auto e = pCaloHit->GetInputEnergy();

        HepEVD::Hit *evdHit = new HepEVD::Hit({x, y, z}, e);
        evdHit->addProperties({{"SemanticLabel", label}});
        hitsToVis.push_back(evdHit);
    }

    HepEVD::getServer()->addHits(hitsToVis);
    HepEVD::saveState("SemanticLabels");
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

void BuildVolumeStitchingEdges(CaloHitList filteredHits, Eigen::MatrixXf &tpcBounds, float maxGap, std::vector<std::pair<int, int>> &edges)
{
    // Map each hit to the volume it belongs to.
    std::map<int, std::vector<int>> hitsToVolumeMap;
    std::vector<CartesianVector> hitPositions;
    hitPositions.reserve(filteredHits.size());

    int hit_idx = 0;
    for (const auto &pCaloHit : filteredHits)
    {
        const auto &pos = pCaloHit->GetPositionVector();
        hitPositions.push_back(pos);

        // Find which volume this hit is inside.
        for (int vol_idx = 0; vol_idx < tpcBounds.cols(); ++vol_idx)
        {
            const auto &bounds = tpcBounds.col(vol_idx);
            const float margin{10.f};
            const bool isInside = (pos.GetX() >= bounds(0) - margin && pos.GetX() <= bounds(1) + margin &&
                pos.GetZ() >= bounds(2) - margin && pos.GetZ() <= bounds(3) + margin);

            if (!isInside)
                continue;

            hitsToVolumeMap[vol_idx].push_back(hit_idx);
            break;
        }
        ++hit_idx;
    }

    // Get a list of all volume indices that contain points.
    std::vector<int> volumeIndices;
    for (const auto &pair : hitsToVolumeMap)
        volumeIndices.push_back(pair.first);
    const float maxDistSquared = (maxGap * 1.5f) * (maxGap * 1.5f);

    // Iterate over unique pairs of volumes to find adjacent ones.
    for (size_t i = 0; i < volumeIndices.size(); ++i)
    {
        for (size_t j = i + 1; j < volumeIndices.size(); ++j)
        {
            const int volAIdx = volumeIndices[i];
            const int volBIdx = volumeIndices[j];

            const auto &boundA = tpcBounds.col(volAIdx);
            const auto &boundsB = tpcBounds.col(volBIdx);

            // Check for adjacency...
            float centerAX, centerAZ, widthAX, widthAZ;
            float centerBX, centerBZ, witdthBZ, widthBZ;
            GetVolumeProps(boundA, centerAX, centerAZ, widthAX, widthAZ);
            GetVolumeProps(boundsB, centerBX, centerBZ, witdthBZ, widthBZ);

            const float gapX = std::max(0.0f, std::abs(centerAX - centerBX) - (widthAX + witdthBZ) / 2.0f);
            const float gapZ = std::max(0.0f, std::abs(centerAZ - centerBZ) - (widthAZ + widthBZ) / 2.0f);

            const bool areAdjacent = (gapX > 0 && gapX < maxGap * 1.5f) || (gapZ > 0 && gapZ < maxGap * 1.5f);

            if (!areAdjacent)
                continue;

            const auto &pointsAIdxs = hitsToVolumeMap.at(volAIdx);
            const auto &pointsBIdxs = hitsToVolumeMap.at(volBIdx);

            if (pointsAIdxs.empty() || pointsBIdxs.empty())
                continue;

            // For each point in A, find its single nearest neighbor in B.
            for (const int idxA : pointsAIdxs)
            {
                float minDistSqd = std::numeric_limits<float>::max();
                int bestIDxB = -1;

                for (const int idxB : pointsBIdxs)
                {
                    const float distSqd = (hitPositions[idxA] - hitPositions[idxB]).GetMagnitudeSquared();
                    if (distSqd < minDistSqd)
                    {
                        minDistSqd = distSqd;
                        bestIDxB = idxB;
                    }
                }

                // Add an edge if the distance to the nearest neighbor is within the threshold.
                if (bestIDxB != -1 && minDistSqd < maxDistSquared)
                    edges.emplace_back(idxA, bestIDxB);
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::GetGraphData(const CaloHitList &caloHits, std::vector<CartesianVector> &pos,
    std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges)
{

    // Build a N, 3 matrix of hit positions...
    CaloHitList filteredHits;
    Eigen::MatrixXf hitMatrix(caloHits.size(), 3);

    for (const auto pCaloHit : caloHits)
    {
        if (nullptr == pCaloHit)
            continue;

        const CartesianVector &hitPos{pCaloHit->GetPositionVector()};
        hitMatrix.row(filteredHits.size()) << hitPos.GetX(), hitPos.GetY(), hitPos.GetZ();
        filteredHits.push_back(pCaloHit);
    }

    HepEVD::addHits(&filteredHits);
    HepEVD::saveState("DLSlicingRawHits");

    // Correct the size of the hit matrix to the number of hits we actually have.
    hitMatrix.conservativeResize(filteredHits.size(), 3);

    // We can now use the hit matrix to create the input for the model.
    const unsigned int k(4);

    // Pre-extract positions to a contiguous vector.
    // This massively helps with speeding up the KNN search.
    const int numHits = hitMatrix.rows();
    std::vector<Eigen::Vector3f> hitPosList(numHits);
    for (int i = 0; i < numHits; ++i) {
        hitPosList[i] = Eigen::Vector3f(hitMatrix(i, 0), hitMatrix(i, 1), hitMatrix(i, 2));
    }

    // Also reserve space for the edges, since we know we will have at least numHits * k edges from the KNN search.
    std::vector<std::pair<int, int>> knnEdges;
    knnEdges.reserve(numHits * k);

    // The Smart Double Loop
    for (int r = 0; r < numHits; ++r) {
        const Eigen::Vector3f& currentPos = hitPosList[r];

        // A max-heap that only ever holds exactly 4 elements
        std::priority_queue<std::pair<float, int>> topK;

        for (int i = 0; i < numHits; ++i) {
            if (r == i) continue;

            const float dist_sq = (currentPos - hitPosList[i]).squaredNorm();

            // If we have less than k neighbors, add it.
            if (topK.size() < k) {
                topK.push({dist_sq, i});
            }
            // If this hit is closer than the furthest hit in our top 4, swap them.
            else if (dist_sq < topK.top().first) {
                topK.pop();
                topK.push({dist_sq, i});
            }
        }

        // Extract the exact 4 nearest neighbors safely
        while (!topK.empty()) {
            knnEdges.emplace_back(r, topK.top().second);
            topK.pop();
        }

        // Update the graph!
        const auto &pCaloHit{*std::next(filteredHits.begin(), r)};
        pos.emplace_back(pCaloHit->GetPositionVector());
        node_features.emplace_back(std::array<float, 1>{pCaloHit->GetInputEnergy()});
    }

    // Append our fast edges to your main edge list
    edges.insert(edges.end(), knnEdges.begin(), knnEdges.end());

    std::cout << "At the end of KNN search, we have " << pos.size() << " nodes and " << edges.size() << " edges." << std::endl;

    // Now, we have a full graph with a KNN structure.
    // We just need to update it to also include so-called "stitching edges".
    // These are edges that connect over the gaps between detector modules.
    // To do that, we just get the detector geometry, and find all the hits that
    // within some distance of the gaps. We can then do a simple distance check
    // to see if the hits are close enough to be connected by an edge.
    const auto &geoManager{this->GetPandora().GetGeometry()};
    Eigen::MatrixXf tpcBounds(4, geoManager->GetLArTPCMap().size());

    for (unsigned int i = 0; i < geoManager->GetLArTPCMap().size(); ++i)
    {
        const auto &tpc{*std::next(geoManager->GetLArTPCMap().begin(), i)};
        const float xCenter{tpc.second->GetCenterX()};
        const float xHalfWidth{tpc.second->GetWidthX() / 2.0f};

        const float zCenter{tpc.second->GetCenterZ()};
        const float zHalfWidth{tpc.second->GetWidthZ() / 2.0f};

        tpcBounds.col(i) << xCenter - xHalfWidth, xCenter + xHalfWidth, zCenter - zHalfWidth, zCenter + zHalfWidth;
    }

    // Iterate over the TPC bounds and find the largest gap between them.
    std::vector<float> xMins, xMaxs, zMins, zMaxs;
    for (unsigned int i = 0; i < tpcBounds.cols(); ++i)
    {
        xMins.push_back(tpcBounds(0, i));
        xMaxs.push_back(tpcBounds(1, i));
        zMins.push_back(tpcBounds(2, i));
        zMaxs.push_back(tpcBounds(3, i));
    }
    std::sort(xMins.begin(), xMins.end());
    std::sort(xMaxs.begin(), xMaxs.end());
    std::sort(zMins.begin(), zMins.end());
    std::sort(zMaxs.begin(), zMaxs.end());

    float maxGap{0.0f};
    for (size_t i = 0; i < xMins.size() - 1; ++i)
    {
        const float xGap = xMins[i + 1] - xMaxs[i];
        if (xGap > maxGap)
            maxGap = xGap;

        const float zGap = zMins[i + 1] - zMaxs[i];
        if (zGap > maxGap)
            maxGap = zGap;
    }

    // Fallback just in case geometry bounds overlap perfectly
    if (maxGap <= 0.0f)
        maxGap = 5.0f;

    std::cout << "Max gap between TPC volumes is " << maxGap << " cm." << std::endl;
    BuildVolumeStitchingEdges(filteredHits, tpcBounds, maxGap, edges);
    std::cout << "After adding stitching edges, we now have " << pos.size() << " nodes and " << edges.size() << " edges." << std::endl;

    // Almost there. We want to coalesce the edges down to remove any duplicates.
    std::set<std::pair<int, int>> uniqueEdges;
    for (const auto &edge : edges)
    {
        if (edge.first == edge.second)
            continue; // No self loops

        uniqueEdges.emplace(edge.first, edge.second);
    }

    edges = std::vector<std::pair<int, int>>(uniqueEdges.begin(), uniqueEdges.end());

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
        // First, we just set the edge indices.
        edgeIndexTensor[0][i] = edges[i].first;
        edgeIndexTensor[1][i] = edges[i].second;

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
        edgeAttrTensor[i][0] = scaledRelX;
        edgeAttrTensor[i][1] = scaledRelY;
        edgeAttrTensor[i][2] = scaledRelZ;
        edgeAttrTensor[i][3] = scaledDist;
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

    // DUMP GRAPH FOR PYTHON COMPARISON
    std::ofstream posFile("cpp_pos.csv");
    for (const auto &p : pos)
        posFile << p.GetX() << "," << p.GetY() << "," << p.GetZ() << "\n";
    std::ofstream edgeFile("cpp_edge_index.csv");
    for (const auto &edge : edges)
        edgeFile << edge.first << "," << edge.second << "\n";

    // DUMP X (NODE FEATURES)
    std::ofstream xFile("cpp_x.csv");
    for (const auto &feat : node_features)
    {
        xFile << feat[0] << "\n";
    }
    xFile.close();

    // DUMP EDGE ATTRIBUTES
    std::ofstream attrFile("cpp_edge_attr.csv");
    for (int i = 0; i < numEdges; ++i)
    {
        // Assuming standard LibTorch tensor access based on your code
        attrFile << edgeAttrTensor[i][0].item<float>() << "," << edgeAttrTensor[i][1].item<float>() << ","
                 << edgeAttrTensor[i][2].item<float>() << "," << edgeAttrTensor[i][3].item<float>() << "\n";
    }
    attrFile.close();

    std::cout << "Finished building graph input for the model." << std::endl;

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

    // INFO: Unlike DLVertexing, this truly does match the size of the m_threshold vector.
    //       This is because we have an extra "background" class, for noise.
    m_nDistanceClasses = m_thresholds.size();

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_dl_content
