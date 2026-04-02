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

#include <mach/mach.h>

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

#define DEBUG_MODE 0
#if DEBUG_MODE
#define HEP_EVD_PANDORA_HELPERS 1
#include "hep_evd.h"
#endif

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

namespace
{

double GetCurrentRssMb()
{
    mach_task_basic_info taskInfo;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;

    const kern_return_t status = task_info(
        mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&taskInfo), &infoCount);

    if (KERN_SUCCESS != status)
        return -1.0;

    return static_cast<double>(taskInfo.resident_size) / (1024.0 * 1024.0);
}

void PrintCurrentRss(const std::string &label)
{
    const double rssMb = GetCurrentRssMb();

    if (rssMb < 0.0)
    {
        std::cout << "[RAM] " << label << ": RSS unavailable" << std::endl;
        return;
    }

    std::cout << "[RAM] " << label << ": RSS " << rssMb << " MB" << std::endl;
}

} // namespace

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
    PrintCurrentRss("Infer/start");

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
    PrintCurrentRss("After GetGraphData");

    LArDLHelper::TorchInputVector inputs;
    t1 = std::chrono::high_resolution_clock::now();
    this->BuildGraph(inputs, nodes, node_features, edges);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Building graph took " << duration << " ms." << std::endl;

    // Free the C++ intermediate containers: they are fully encoded in tensors now.
    node_features.clear();
    node_features.shrink_to_fit();
    edges.clear();
    edges.shrink_to_fit();
    PrintCurrentRss("After BuildGraph + vector release");

    LArDLHelper::TorchMultiOutput semanticOutput;
    t1 = std::chrono::high_resolution_clock::now();
    LArDLHelper::Forward(m_modelFile, inputs, semanticOutput);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Inference took " << duration << " ms." << std::endl;
    PrintCurrentRss("After semantic inference");

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
    const auto &outputTuple = semanticOutput.toTuple();
    const auto &semanticLabels{outputTuple->elements()[0].toTensor()};
    const auto &rawEmbeddings{outputTuple->elements()[1].toTensor()};
    const auto &posEmbeddings{outputTuple->elements()[2].toTensor()};

    // Lets do some basic checks...
    std::cout << "Semantic Labels: " << semanticLabels.sizes() << ", " << semanticLabels.dtype() << std::endl;
    std::cout << "Raw Embeddings: " << rawEmbeddings.sizes() << ", " << rawEmbeddings.dtype() << std::endl;
    std::cout << "Pos Embeddings: " << posEmbeddings.sizes() << ", " << posEmbeddings.dtype() << std::endl;

#if DEBUG_MODE
    // DEBUG: Add visualization of the semantic labels to EVD, to check they
    // look sensible before we try to do any more complicated processing.
    const auto argMaxLabels = torch::argmax(semanticLabels, 1);
    HepEVD::Hits hitsToVis;

    int evdHitIdx{0};
    for (const auto pCaloHit : *pCaloHitList)
    {
        if (nullptr == pCaloHit)
            continue;

        const double label = argMaxLabels[evdHitIdx].item<double>();

        const auto x = pCaloHit->GetPositionVector().GetX();
        const auto y = pCaloHit->GetPositionVector().GetY();
        const auto z = pCaloHit->GetPositionVector().GetZ();
        const auto e = pCaloHit->GetInputEnergy();

        HepEVD::Hit *evdHit = new HepEVD::Hit({x, y, z}, e);
        evdHit->addProperties({{"SemanticLabel", label}});

        if (label <= 2)
            evdHit->addProperties({{"SeedCandidate", 1}});

        hitsToVis.push_back(evdHit);

        evdHitIdx++;
    }

    HepEVD::setHepEVDGeometry(this->GetPandora().GetGeometry());
    HepEVD::getServer()->addHits(hitsToVis);
#endif

    // Next, process the semantic labels with the Hough Transform to find vertex
    // candidates. Scope the working buffers so they're freed before instance seg.
    std::vector<CartesianVector> foundVertices;
    {
        const unsigned int numHits = semanticLabels.size(0);
        const auto contiguousSemanticLabels = semanticLabels.contiguous();
        std::vector<float> semanticLabelsVec(
            contiguousSemanticLabels.data_ptr<float>(), contiguousSemanticLabels.data_ptr<float>() + (numHits * m_nDistanceClasses));

        // Setup and run the Hough Transform vertex finder.
        t1 = std::chrono::high_resolution_clock::now();
        FastHoughFinder houghFinder(m_thresholds, m_scalingFactor);
        foundVertices = houghFinder.Fit(nodes, semanticLabelsVec);
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "Hough Transform vertex finding took " << duration << " ms." << std::endl;
        std::cout << "Found " << foundVertices.size() << " vertex candidates." << std::endl;
    } // semanticLabelsVec freed here.

    // nodes is no longer needed after the Hough finder.
    nodes.clear();
    nodes.shrink_to_fit();
    PrintCurrentRss("After Hough + node release");

#if DEBUG_MODE
    // DEBUG: Add them to HepEVD.
    HepEVD::Markers pointsToVis;
    for (const auto &vertex : foundVertices)
    {
        HepEVD::Point *evdPoint = new HepEVD::Point({vertex.GetX(), vertex.GetY(), vertex.GetZ()});
        pointsToVis.push_back(*evdPoint);
    }

    HepEVD::getServer()->addMarkers(pointsToVis);
    HepEVD::saveState("FoundVertices");
#endif

    // Start setting up the inputs for instance segmentation.
    const int numCandidates = foundVertices.size();

    if (numCandidates == 0)
    {
        std::cout << "DLSlicingAlgorithm::Infer - no vertex candidates found, skipping instance segmentation step" << std::endl;
        // TODO: What to do here?
        return STATUS_CODE_SUCCESS;
    }

    const auto asFloat = torch::TensorOptions().dtype(torch::kFloat32);
    LArDLHelper::TorchInput candidateTensor;
    LArDLHelper::InitialiseInput({numCandidates, 3}, candidateTensor, asFloat);
    float *candidateTensorData = candidateTensor.data_ptr<float>();

    // Populate the candidate tensor with the positions of the found vertices.
    for (unsigned int i = 0; i < numCandidates; ++i)
    {
        candidateTensorData[i * 3 + 0] = foundVertices[i].GetX();
        candidateTensorData[i * 3 + 1] = foundVertices[i].GetY();
        candidateTensorData[i * 3 + 2] = foundVertices[i].GetZ();
    }

    // Now, populate the full input tensor with all the required data:
    // 1) The semantic distance logits for each hit.
    // 2) The raw embeddings for each hit.
    // 3) The position embeddings for each hit.
    // 4) The candidate vertex positions.
    // 5) The edges between hits.
    const auto edgeIndexTensor = inputs[2].toTensor();
    inputs.clear(); // Free xTensor, posTensor, edgeAttrTensor, batchTensor.
    LArDLHelper::TorchInputVector fullInputTensor;
    fullInputTensor.push_back(semanticLabels);
    fullInputTensor.push_back(rawEmbeddings);
    fullInputTensor.push_back(posEmbeddings);
    fullInputTensor.push_back(candidateTensor);
    fullInputTensor.push_back(edgeIndexTensor);
    PrintCurrentRss("After building instance inputs");

    // Okay, we are good to go!
    t1 = std::chrono::high_resolution_clock::now();
    LArDLHelper::TorchOutput instanceOutput;
    LArDLHelper::Forward(m_modelFile, fullInputTensor, instanceOutput, "predict_instances");
    fullInputTensor.clear();
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Instance segmentation inference took " << duration << " ms." << std::endl;
    PrintCurrentRss("After instance inference");

    std::cout << "DLSlicingAlgorithm::Infer - instance segmentation output: " << instanceOutput.sizes() << ", " << instanceOutput.dtype() << std::endl;
    const auto instancePreds = std::get<1>(torch::max(torch::sigmoid(instanceOutput), 1));
    instanceOutput = at::Tensor(); // Free the raw model output now that we have the predictions.
    PrintCurrentRss("After instance output release");

#if DEBUG_MODE
    // DEBUG: Visualise the pre-post-processing clusters.
    std::map<int, HepEVD::Hits> instanceHitsMap;
    evdHitIdx = 0;

    for (const auto pCaloHit : *pCaloHitList)
    {
        if (nullptr == pCaloHit)
            continue;

        const auto x = pCaloHit->GetPositionVector().GetX();
        const auto y = pCaloHit->GetPositionVector().GetY();
        const auto z = pCaloHit->GetPositionVector().GetZ();
        const auto e = pCaloHit->GetInputEnergy();

        const auto clusterPrediction = instancePreds[evdHitIdx].item<int>();

        HepEVD::Hit *evdHit = new HepEVD::Hit({x, y, z}, e);
        instanceHitsMap[clusterPrediction].push_back(evdHit);

        ++evdHitIdx;
    }

    // DEBUG: Flatten to particles.
    HepEVD::Particles particlesToVis;
    unsigned int clusterId = 0;
    for (const auto &[clusterId, hits] : instanceHitsMap)
    {
        std::cout << "Cluster " << clusterId << ": " << hits.size() << " hits" << std::endl;
        HepEVD::Particle *evdParticle = new HepEVD::Particle(hits, std::to_string(clusterId));
        particlesToVis.push_back(evdParticle);
    }

    HepEVD::getServer()->addParticles(particlesToVis);
    HepEVD::saveState("Slicing Result");
    HepEVD::startServer();
#endif

    // For now...lets just write out a new Cluster list, with one cluster per
    // predicted instance.
    // This will likely eventually need to be made into a LArRecoND algorithm,
    // that is based on EventSlicingThreeD, but this will work for now
    // and we can have a basic LArRecoND tool that just loads this cluster list.
    std::string clusterListName = m_outputClusterListName;
    const ClusterList *pClusterList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pClusterList, clusterListName));

    // Build the clusters from the predicted instances.
    std::map<int, std::list<const CaloHit *>> clusterHitsMap;
    int hitIdx{0};

    for (const auto pCaloHit : *pCaloHitList)
    {
        if (nullptr == pCaloHit)
            continue;

        const auto clusterPrediction = instancePreds[hitIdx].item<int>();
        clusterHitsMap[clusterPrediction].push_back(pCaloHit);

        ++hitIdx;
    }

    for (const auto &[clusterId, hits] : clusterHitsMap)
    {
        if (hits.empty())
            continue;

        PandoraContentApi::Cluster::Parameters clusterParameters;
        clusterParameters.m_caloHitList = hits;
        const Cluster *pCluster(nullptr);
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, clusterParameters, pCluster));
    }

    if (!pClusterList->empty())
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_outputClusterListName));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Cluster>(*this, m_outputClusterListName));
    }

    PrintCurrentRss("Infer/end");

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
    const auto &gapList = this->GetPandora().GetGeometry()->GetDetectorGapList();
    if (gapList.empty())
    {
        std::cout << "DLSlicingAlgorithm::BuildVolumeStitchingEdges - no gaps found, skipping stitching." << std::endl;
        return;
    }

    // Distance from gap boundary to consider a hit
    // TODO: Param?
    const float margin = 5.0f;

    for (const auto *const pGap : gapList)
    {
        const pandora::BoxGap *const pBoxGap = dynamic_cast<const pandora::BoxGap *>(pGap);

        if (!pBoxGap)
            continue;

        const pandora::CartesianVector &vertex = pBoxGap->GetVertex();
        const pandora::CartesianVector &side1 = pBoxGap->GetSide1();
        const pandora::CartesianVector &side2 = pBoxGap->GetSide2();
        const pandora::CartesianVector &side3 = pBoxGap->GetSide3();

        // Find the absolute center of the gap
        const float cX = vertex.GetX() + (side1.GetX() * 0.5f);
        const float cY = vertex.GetY() + (side2.GetY() * 0.5f);
        const float cZ = vertex.GetZ() + (side3.GetZ() * 0.5f);

        // Full widths of the gap volume
        const float wX = std::fabs(side1.GetX());
        const float wY = std::fabs(side2.GetY());
        const float wZ = std::fabs(side3.GetZ());

        // Determine the gap's separation axis
        const bool isXGap = (wX < wY && wX < wZ);
        const bool isZGap = (wZ < wX && wZ < wY);

        if (!isXGap && !isZGap)
            continue;

        // Scale the connection radius dynamically based on the width of this specific gap
        const float gapThickness = isXGap ? wX : wZ;
        const float max_dist_sq = (gapThickness * 1.5f) * (gapThickness * 1.5f);

        std::vector<int> sideA, sideB;

        // Fast AABB check to find hits near this specific gap
        for (size_t i = 0; i < pos.size(); ++i)
        {
            const float dx = std::max(0.0f, std::fabs(pos[i].GetX() - cX) - (wX * 0.5f));
            const float dy = std::max(0.0f, std::fabs(pos[i].GetY() - cY) - (wY * 0.5f));
            const float dz = std::max(0.0f, std::fabs(pos[i].GetZ() - cZ) - (wZ * 0.5f));

            if (dx <= margin && dy <= margin && dz <= margin)
            {
                if (isXGap)
                {
                    if (pos[i].GetX() < cX) sideA.push_back(i);
                    else sideB.push_back(i);
                }
                else // isZGap
                {
                    if (pos[i].GetZ() < cZ) sideA.push_back(i);
                    else sideB.push_back(i);
                }
            }
        }

        if (sideA.empty() || sideB.empty())
            continue;

        // Build temporary KD-Trees for each face of the gap
        std::vector<KnnKdTree::KnnNode> nodesA, nodesB;
        nodesA.reserve(sideA.size());
        nodesB.reserve(sideB.size());

        for (const int idx : sideA) {
            nodesA.push_back({{pos[idx].GetX(), pos[idx].GetY(), pos[idx].GetZ()}, idx});
        }
        for (const int idx : sideB) {
            nodesB.push_back({{pos[idx].GetX(), pos[idx].GetY(), pos[idx].GetZ()}, idx});
        }

        KnnKdTree treeA(nodesA);
        KnnKdTree treeB(nodesB);

        // INFO: Technically tunable...but setting this too high can lead to a
        // big imbalance between KNN edges and stitching edges.
        const int greedyK{1};

        // Search from Side A -> Side B
        for (const int idxA : sideA)
        {
            KnnKdTree::KnnNode nodeA = {{pos[idxA].GetX(), pos[idxA].GetY(), pos[idxA].GetZ()}, idxA};
            std::vector<int> neighborsInB = treeB.FindNearestNeighbours(nodeA, greedyK);

            for (const int idxB : neighborsInB)
            {
                if ((pos[idxA] - pos[idxB]).GetMagnitudeSquared() < max_dist_sq)
                {
                    edges.push_back({idxA, idxB});
                }
            }
        }

        // And the inverse...
        for (const int idxB : sideB)
        {
            KnnKdTree::KnnNode nodeB = {{pos[idxB].GetX(), pos[idxB].GetY(), pos[idxB].GetZ()}, idxB};
            std::vector<int> neighborsInA = treeA.FindNearestNeighbours(nodeB, greedyK);

            for (const int idxA : neighborsInA)
            {
                if ((pos[idxB] - pos[idxA]).GetMagnitudeSquared() < max_dist_sq)
                {
                    edges.push_back({idxB, idxA});
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
            // INFO: KNN Edges are directed. I.e. A -> B does not imply B -> A.
            edges.push_back({node.original_id, neighbourIdx});
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

    // torch::empty avoids a redundant zero-fill since every element is overwritten below.
    LArDLHelper::TorchInput posTensor, xTensor, edgeIndexTensor, edgeAttrTensor;
    posTensor       = torch::empty({numNodes, 3},         asFloat);
    edgeIndexTensor = torch::empty({2, numEdges},         asInt);
    xTensor         = torch::empty({numNodes, numFeatures}, asFloat);
    edgeAttrTensor  = torch::empty({numEdges, edgeShape}, asFloat);

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
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputClusterListName", m_outputClusterListName));

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "DistanceThresholds", m_thresholds));

    m_nDistanceClasses = m_thresholds.size() + 1; // We have one more class than thresholds, as the thresholds define the boundaries between classes.

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_dl_content
