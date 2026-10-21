/**
 *  @file   larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning slicing algorithm.
 *
 *  $Log: $
 */

#include <string>
#include <vector>

#include "Api/PandoraApi.h"
#include "Objects/CartesianVector.h"
#include "Pandora/PandoraInternal.h"
#include "Pandora/StatusCodes.h"

#include "larpandoracontent/LArHelpers/LArFileHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

#include "larpandoradlcontent/LArSlicing/DlSlicingAlgorithm.h"
#include "larpandoradlcontent/LArSlicing/HoughFinder.h"

#include <Eigen/Dense>
#include <c10/core/TensorOptions.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlSlicingAlgorithm::DlSlicingAlgorithm() :
    m_scalingFactor{-1.0f},
    m_thresholds{},
    m_nDistanceClasses{-1},
    m_runPostProcessing{false}
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
    const unsigned int numHits = pCaloHitList->size();

    std::vector<CartesianVector> nodes;
    std::vector<std::array<float, 1>> node_features;
    this->GetNodeData(*pCaloHitList, nodes, node_features);

    LArDLHelper::TorchInputVector inputs;
    this->BuildInput(inputs, nodes, node_features);

    node_features.clear();
    node_features.shrink_to_fit();

    torch::Tensor instancePreds;
    torch::Tensor fullInstancePreds;

    // Data structures for post-processing...
    std::vector<CartesianVector> foundVertices;
    std::vector<int> candidateIndices;
    std::vector<int> noiseMask(numHits, 0);

    {
        LArDLHelper::TorchMultiOutput semanticOutput;
        LArDLHelper::Forward(m_modelFile, inputs, semanticOutput);

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
        // 3) The encoded positions, same as above. Just saves re-computing them.
        const auto semanticLabels = semanticOutput.toTuple()->elements()[0].toTensor();
        const auto hitEmbeddings = semanticOutput.toTuple()->elements()[1].toTensor();
        const auto encodedPos = semanticOutput.toTuple()->elements()[2].toTensor();
        semanticOutput = LArDLHelper::TorchMultiOutput();

        // Next, process the semantic labels with the Hough Transform to find vertex
        // candidates. Scope the working buffers so they're freed before instance seg.
        {
            const auto contiguousSemanticLabels = semanticLabels.contiguous();
            std::vector<float> semanticLabelsVec(
                contiguousSemanticLabels.data_ptr<float>(), contiguousSemanticLabels.data_ptr<float>() + (numHits * m_nDistanceClasses));

            // Setup and run the Hough Transform vertex finder.
            ThreeDVertexingHoughTransform houghFinder(m_thresholds, m_scalingFactor);
            foundVertices = houghFinder.Fit(nodes, semanticLabelsVec);
        }

        // Store some info about the candidates for later post processing...
        for (int i = 0; i < numHits; ++i)
        {
            for (const auto &vtx : foundVertices)
            {
                if ((nodes[i] - vtx).GetMagnitudeSquared() < 100.f)
                {
                    candidateIndices.push_back(i);
                    break;
                }
            }
        }

        // Similarly, pull out a noise mask before we clear that.
        const int noiseClass = m_nDistanceClasses - 1;
        auto semanticPreds = torch::argmax(semanticLabels, 1);
        for (int i = 0; i < numHits; ++i)
        {
            if (semanticPreds[i].item<int>() == noiseClass)
            {
                noiseMask[i] = 1;
            }
        }

        // nodes is no longer needed after the Hough finder.
        nodes.clear();
        nodes.shrink_to_fit();

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
        // 1) The position embeddings for each hit.
        // 2) The raw embeddings for each hit.
        // 3) The semantic distance logits for each hit.
        // 4) The candidate vertex positions.
        inputs.clear();
        LArDLHelper::TorchInputVector fullInputTensor;
        fullInputTensor.push_back(std::move(encodedPos));
        fullInputTensor.push_back(std::move(hitEmbeddings));
        fullInputTensor.push_back(std::move(semanticLabels));
        fullInputTensor.push_back(std::move(candidateTensor));

        // Okay, we are good to go!
        LArDLHelper::TorchOutput instanceOutput;
        LArDLHelper::Forward(m_modelFile, fullInputTensor, instanceOutput, "predict_instances");
        fullInputTensor.clear();

        instancePreds = std::get<1>(torch::max(torch::sigmoid(instanceOutput), 1));
        fullInstancePreds = torch::sigmoid(instanceOutput);
        instanceOutput = at::Tensor();
    }

    // Final cluster labels, to be updated by the post-processing if ran.
    std::vector<int> finalLabels(instancePreds.size(0), -1);

    for (int i = 0; i < numHits; ++i)
    {
        int label = instancePreds[i].item<int>();
        finalLabels[i] = (noiseMask[i] == 1) ? -1 : label;
    }

    // Build the final cluster -> hit map, so we can save it.
    std::map<int, std::list<const CaloHit *>> clusterHitsMap;
    int hitIdx = 0;
    for (const auto pCaloHit : *pCaloHitList)
    {
        if (nullptr == pCaloHit)
            continue;

        const auto clusterPrediction = finalLabels[hitIdx];
        if (clusterPrediction >= 0)
            clusterHitsMap[clusterPrediction].push_back(pCaloHit);

        ++hitIdx;
    }

    // For now...lets just write out a new Cluster list, with one cluster per
    // predicted instance.
    // This will likely eventually need to be made into a LArRecoND algorithm,
    // that is based on EventSlicingThreeD, but this will work for now
    // and we can have a basic LArRecoND tool that just loads this cluster list.
    const ClusterList *pClusterList(nullptr);
    std::string temporaryListName;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pClusterList, temporaryListName));

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

    // We should also write out the predicted vertices to a new output list, as
    // they may be useful for seeding later algorithms, even if the instance
    // segmentation fails.
    const VertexList *pVertexList(nullptr);
    temporaryListName.clear();
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pVertexList, temporaryListName));

    for (const auto &vertex : foundVertices)
    {
        PandoraContentApi::Vertex::Parameters parameters;
        parameters.m_position = vertex;
        parameters.m_vertexLabel = VERTEX_INTERACTION;
        parameters.m_vertexType = VERTEX_3D;

        const Vertex *pVertex(nullptr);
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Vertex::Create(*this, parameters, pVertex));
    }

    if (!pVertexList->empty())
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Vertex>(*this, m_outputVertexListName));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Vertex>(*this, m_outputVertexListName));
    }

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::GetNodeData(const CaloHitList &caloHits, std::vector<CartesianVector> &pos, std::vector<std::array<float, 1>> &node_features)
{
    pos.reserve(caloHits.size());
    node_features.reserve(caloHits.size());

    // Populate the positional and node feature vectors from the CaloHits.
    int hitIdx{0};
    for (const auto pCaloHit : caloHits)
    {
        if (nullptr == pCaloHit)
            continue;

        const CartesianVector &hitPos = pCaloHit->GetPositionVector();

        pos.push_back(hitPos);
        node_features.push_back({pCaloHit->GetInputEnergy() / 10.f});
    }

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlSlicingAlgorithm::BuildInput(
    LArDLHelper::TorchInputVector &inputs, std::vector<CartesianVector> &pos, std::vector<std::array<float, 1>> &node_features)
{
    const int numNodes{static_cast<int>(pos.size())};
    const int numFeatures{static_cast<int>(node_features[0].size())};

    LArDLHelper::TorchInput posTensor, xTensor;
    posTensor = torch::empty({numNodes, 3}, torch::kFloat32);
    xTensor = torch::empty({numNodes, numFeatures}, torch::kFloat32);

    // INFO: Fake batch tensor (i.e. our batch size is 1 in inference)
    torch::Tensor batchTensor = torch::zeros(numNodes, torch::kLong);

    float *posTensorPtr = posTensor.data_ptr<float>();
    float *xTensorPtr = xTensor.data_ptr<float>();

    // Fill in the position and node feature tensors...
    for (int i = 0; i < numNodes; ++i)
    {
        posTensorPtr[i * 3 + 0] = pos[i].GetX();
        posTensorPtr[i * 3 + 1] = pos[i].GetY();
        posTensorPtr[i * 3 + 2] = pos[i].GetZ();
        xTensorPtr[i] = node_features[i][0];
    }

    // Finally, stick them together into the input vector.
    inputs.insert(inputs.end(), {xTensor, posTensor, batchTensor});

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
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputVertexListName", m_outputVertexListName));

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "DistanceThresholds", m_thresholds));

    m_nDistanceClasses = m_thresholds.size() + 1; // We have one more class than thresholds, as the thresholds define the boundaries between classes.

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_dl_content
