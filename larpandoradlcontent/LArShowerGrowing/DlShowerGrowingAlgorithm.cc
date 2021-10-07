/**
 *  @file   larpandoradlcontent/LArShowerGrowing/DlShowerGrowingAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning shower growing algorithm.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include <torch/script.h>
#include <torch/torch.h>

#include "larpandoradlcontent/LArShowerGrowing/DlShowerGrowingAlgorithm.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArFileHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArMvaHelper.h"
#include "larpandoracontent/LArHelpers/LArVertexHelper.h"

#include "larpandoracontent/LArObjects/LArCaloHit.h"
#include "larpandoracontent/LArObjects/LArTwoDSlidingFitResult.h"

#include <chrono>
#include <fstream>

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlShowerGrowingAlgorithm::DlShowerGrowingAlgorithm() :
    m_visualize(false), m_useTrainingMode(false), m_limitedEdges(false), m_trainingOutputFile("")
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

DlShowerGrowingAlgorithm::~DlShowerGrowingAlgorithm()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlShowerGrowingAlgorithm::Run()
{
    if (m_useTrainingMode)
        return this->Train();
    else
        return this->Infer();
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlShowerGrowingAlgorithm::Train()
{
    for (const std::string &listName : m_clusterListNames)
    {

        const ClusterList *pClusterList = nullptr;

        try
        {
            PANDORA_RETURN_RESULT_IF_AND_IF(
                STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, listName, pClusterList));
        }
        catch (StatusCodeException e)
        {
            std::cout << "Failed to get cluster list: " << e.ToString() << std::endl;
            continue;
        }

        if (pClusterList == nullptr || pClusterList->size() == 0)
        {
            std::cout << "Cluster list was empty." << std::endl;
            continue;
        }

        this->ProduceTrainingFile(pClusterList, listName);
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlShowerGrowingAlgorithm::Infer()
{
    for (const std::string &listName : m_clusterListNames)
    {

        const ClusterList *pClusterList = nullptr;

        try
        {
            PANDORA_RETURN_RESULT_IF_AND_IF(
                STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, listName, pClusterList));
        }
        catch (StatusCodeException e)
        {
            std::cout << "Failed to get cluster list: " << e.ToString() << std::endl;
            continue;
        }

        if (pClusterList == nullptr || pClusterList->size() == 0)
        {
            std::cout << "Cluster list was empty." << std::endl;
            continue;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        this->InferForView(pClusterList, listName);
        auto t2 = std::chrono::high_resolution_clock::now();

        /* Getting number of milliseconds as an integer. */
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "It took " << ms_int.count() << " milliseconds to run for input " << listName << std::endl;
    }

    if (m_visualize)
        PANDORA_MONITORING_API(ViewEvent(this->GetPandora()));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <typename Func>
struct lambda_as_visitor_wrapper : Func
{
    lambda_as_visitor_wrapper(const Func &f) : Func(f)
    {
    }
    template <typename S, typename I>
    void init(const S &v, I i, I j)
    {
        return Func::operator()(v, i, j);
    }
};

template <typename Mat, typename Func>
void visit_lambda(const Mat &m, const Func &f)
{
    lambda_as_visitor_wrapper<Func> visitor(f);
    m.visit(visitor);
}

StatusCode DlShowerGrowingAlgorithm::InferForView(const ClusterList *clusters, const std::string &listName)
{
    if (clusters->size() <= 1)
        return STATUS_CODE_SUCCESS;

    std::cout << "Infering for view " << listName << std::endl;
    const VertexList *pVertexList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));

    if (pVertexList == nullptr || pVertexList->size() == 0)
        return STATUS_CODE_NOT_FOUND;

    // TODO: Check! Is there more than 1 vertex?
    const Vertex *pVertex = pVertexList->front();
    ClusterList currentClusters(*clusters);

    int runNumber = 0;

    while (runNumber < 10)
    {
        int totalHits = 0;

        std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
        std::cout << "Run " << runNumber << ", with " << currentClusters.size() << " clusters" << std::endl;

        IdClusterMap nodeToCluster;
        NodeFeatureVector nodes;
        EdgeVector edges;
        EdgeFeatureVector edgeFeatures;
        std::cout << "Getting graph data..." << std::endl;
        this->GetGraphData(currentClusters, pVertex, nodeToCluster, nodes, edges, edgeFeatures);

        std::cout << "Picking input cluster..." << std::endl;
        std::vector<std::pair<float, const Cluster *>> clustersToUse;
        for (auto cluster : *clusters)
        {
            if (cluster->GetParticleId() == 13)
                continue;

            float trackProb = 0.f, showerProb = 0.f;
            const float clusterSize = cluster->GetNCaloHits();
            totalHits += clusterSize;

            float xMin = 0.f, xMax = 0.f;
            float zMin = 0.f, zMax = 0.f;
            cluster->GetClusterSpanX(xMin, xMax);
            cluster->GetClusterSpanZ(xMin, xMax, zMin, zMax);
            const float xLen = std::abs(xMax - xMin);
            const float zLen = std::abs(zMax - zMin);
            const float area = xLen * zLen;

            if (LArClusterHelper::GetTrackShowerProbability(cluster, trackProb, showerProb) == STATUS_CODE_SUCCESS)
            {
                const float score = ((showerProb / clusterSize) - (trackProb / clusterSize)) * area;
                clustersToUse.push_back({score, cluster});
            }
            else
                clustersToUse.push_back({clusterSize, cluster});
        }
        std::stable_sort(clustersToUse.begin(), clustersToUse.end());
        std::cout << "Worst Score: " << clustersToUse.front() << std::endl;
        std::cout << "Best Score: " << clustersToUse.back() << std::endl;
        const Cluster *inputCluster(clustersToUse[clustersToUse.size() - runNumber - 1].second);

        LArDLHelper::TorchInputVector inputs;
        std::cout << "Building graph..." << std::endl;
        this->BuildGraph(inputCluster, nodes, edges, edgeFeatures, inputs);

        LArDLHelper::TorchOutput output;
        LArDLHelper::TorchModel &model{m_modelU};

        std::cout << "Starting model!" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        LArDLHelper::Forward(model, inputs, output);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "It took " << ms_int.count() << " milliseconds to run inference" << std::endl;

        const int numberOfClustersStart = currentClusters.size();
        if (this->GrowClusters(listName, inputCluster, nodeToCluster, output, currentClusters) != STATUS_CODE_SUCCESS)
            break;
        const int numberOfClustersEnd = currentClusters.size();

        int remainingHits = 0;
        for (auto cluster : currentClusters)
            if (cluster->GetParticleId() == 11)
                remainingHits += cluster->GetNCaloHits();

        int numberOfHitsAdded = totalHits - remainingHits;

        if (remainingHits <= (totalHits * 0.1) || numberOfClustersStart == numberOfClustersEnd || numberOfHitsAdded <= (totalHits * 0.1))
            break;

        if (m_visualize && listName == "ShowerClustersW")
            this->Visualize(inputs[0].toTensor(), inputs[1].toTensor(), output, listName);

        ++runNumber;
        std::cout << "End of run " << runNumber << ", there are " << currentClusters.size() << " clusters remaining" << std::endl;
        std::cout << "We have " << remainingHits << " hits left, out of  " << totalHits << std::endl;
        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    }

    std::cout << "Finished growing " << listName << "! It took " << runNumber << " runs." << std::endl;

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DlShowerGrowingAlgorithm::GetGraphData(const pandora::ClusterList &clusters, const pandora::Vertex *vertex,
    IdClusterMap &nodeToCluster, NodeFeatureVector &nodes, EdgeVector &edges, EdgeFeatureVector &edgeFeatures)
{
    // For every cluster, round all the hits to the nearest 2.
    // Then, build up the required node features and store them.
    for (auto cluster : clusters)
    {
        if (std::abs(cluster->GetParticleId()) == MU_MINUS)
            continue;

        std::map<std::pair<int, int>, RoundedClusterInfo> roundedClusters;
        CartesianPointVector allCaloHitsForCluster;

        // Pull out all the calo hits that make up this cluster, and store
        // them. Either as a new node, or as part of an existing rounded node.
        for (auto hitList : cluster->GetOrderedCaloHitList())
        {
            for (auto caloHit : *hitList.second)
            {
                allCaloHitsForCluster.push_back(caloHit->GetPositionVector());
                const float x = caloHit->GetPositionVector().GetX();
                const float z = caloHit->GetPositionVector().GetZ();
                const int roundedX = (x / m_rounding) * m_rounding;
                const int roundedZ = (z / m_rounding) * m_rounding;
                std::pair<int, int> roundedPos = {roundedX, roundedZ};

                // INFO: If this rounded hit lies with another rounded hit, store them together.
                if (roundedClusters.count(roundedPos) == 0)
                {
                    Eigen::MatrixXf hits(2, 10);
                    hits.col(0) << x, z;
                    roundedClusters.insert({roundedPos, {hits, 1, x, z}});
                }
                else
                {
                    RoundedClusterInfo &info = roundedClusters[roundedPos];

                    // INFO: The matrix may need resizing. It will be shrunk to its real size later.
                    if (info.numOfHits >= info.hits.cols())
                        info.hits.conservativeResize(Eigen::NoChange, info.numOfHits + 10);

                    info.hits.col(info.numOfHits) << x, z;
                    info.totalX += x;
                    info.totalZ += z;
                    info.numOfHits += 1;
                }
            }
        }

        // INFO: Build a sliding fit to get out a direction for the cluster.
        //       Crucially, this is a per-cluster feature! Each of the sub-nodes
        //       that a cluster is split in to are likely too small to have a reasonable
        //       definition of direction.
        const float orientation(LArVertexHelper::GetClusterDirectionInZ(this->GetPandora(), vertex, cluster, 1.732f, 0.333f));
        const float slidingFitPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
        const int slidingFitWindow = 100;
        CartesianVector direction(0.f, 0.f, 0.f);

        // TODO: Check this! Its cutting off 1 hit "clusters".
        if (allCaloHitsForCluster.size() > 2)
        {
            const TwoDSlidingFitResult fit(&allCaloHitsForCluster, slidingFitWindow, slidingFitPitch);
            direction = fit.GetAxisDirection();
        }
        else if (allCaloHitsForCluster.size() == 2)
            direction = allCaloHitsForCluster[1] - allCaloHitsForCluster[0];
        else
            continue;

        // INFO: Turn the rounded nodes into actual feature vectors.
        for (auto node : roundedClusters)
        {
            RoundedClusterInfo &info = node.second;

            // INFO: Due to resizing, the matrix may be too large, so resize to real size.
            info.hits.conservativeResize(Eigen::NoChange, info.numOfHits);

            const float numHits = info.numOfHits;
            const float xMean = info.totalX / numHits;
            const float zMean = info.totalZ / numHits;
            const float vertexDisplacement = (xMean - vertex->GetPosition().GetX()) + (zMean - vertex->GetPosition().GetZ());

            // Store the node features
            const NodeFeature features = {cluster, info.hits, direction, numHits, orientation, xMean, zMean, vertexDisplacement};
            nodes.push_back(features);
            nodeToCluster[nodes.size() - 1] = cluster;
        }
    }

    if (nodes.size() == 0)
        return;

    // Build up a (2, N) matrix of positions.
    // This matrix can be used to find the KNN to each node.
    Eigen::MatrixXf allNodePositions(2, nodes.size());

    for (unsigned int i = 0; i < nodes.size(); ++i)
        allNodePositions.col(i) << nodes[i].xMean, nodes[i].zMean;

    for (unsigned int currentNode = 0; currentNode < nodes.size(); ++currentNode)
    {

        const auto nodeFeature = nodes[currentNode];
        const auto currentCluster = nodeFeature.cluster;
        Eigen::VectorXf meanPos(2);
        meanPos << nodeFeature.xMean, nodeFeature.zMean;

        std::vector<MatrixIndex> indices(m_kNN, {-1, -1});
        std::vector<float> values(m_kNN, std::numeric_limits<double>::max());

        // INFO: Iterate over the matrix, finding the K nearest nodes.
        visit_lambda((allNodePositions.colwise() - meanPos).cwiseAbs().colwise().squaredNorm(),
            [&](double v, int row, int col)
            {
                const bool isPartOfCurrentCluster = nodes[col].cluster == currentCluster;

                if (!isPartOfCurrentCluster && v < values[0] && v < m_distanceCutOff)
                {
                    const auto it = std::lower_bound(values.rbegin(), values.rend(), v);
                    const int index = std::distance(begin(values), it.base()) - 1;

                    values[index] = v;
                    indices[index] = {row, col};
                }
                else if (!m_limitedEdges && isPartOfCurrentCluster && (int)currentNode != col)
                {
                    edges.push_back({(int)currentNode, col});
                    edgeFeatures.push_back({1.f, 0.f, 0.f, 0.f});
                }
            });

        // INFO: Two options for "internal edges" (i.e. edges between nodes of the same cluster).
        //       Fully connected or only to the next node along.
        //       Limited edges means next node along, as it can drastically cut the number of edges.
        if (m_limitedEdges && currentNode > 0 && nodes[currentNode - 1].cluster == currentCluster)
        {
            edges.push_back({(int)currentNode, (int)currentNode - 1});
            edgeFeatures.push_back({1.f, 0.f, 0.f, 0.f});
        }

        if (m_limitedEdges && currentNode < nodes.size() && nodes[currentNode + 1].cluster == currentCluster)
        {
            edges.push_back({(int)currentNode, (int)currentNode + 1});
            edgeFeatures.push_back({1.f, 0.f, 0.f, 0.f});
        }

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // WARN: Its possible to have less than 5 neighbours, so stop when reaching edges
            //       that are uninitialized.
            if (indices[i].row == -1)
                break;

            const int otherNodeId = indices[i].col;
            const auto otherFeatures = nodes[otherNodeId];
            float closestApproach = std::numeric_limits<double>::max();

            const float angleBetween = nodeFeature.direction.GetOpeningAngle(otherFeatures.direction);
            const float centreDist = (nodeFeature.xMean - otherFeatures.xMean) + (nodeFeature.xMean + otherFeatures.zMean);

            const bool isCloseToVertex =
                nodeFeature.vertexDisplacement < m_vertexProtectionRadius || otherFeatures.vertexDisplacement < m_vertexProtectionRadius;
            const bool isSteepAngle = angleBetween > m_vertexProtectionAngle;

            // INFO: Protect the vertex and be strict about what edges can be made there.
            if (isCloseToVertex && isSteepAngle)
                continue;

            // INFO: Compare every hit in the current node against every hit in the other node.
            //       This way we can find the closest approach between the two nodes.
            for (unsigned int j = 0; j < nodeFeature.hits.cols(); ++j)
            {
                auto hit = nodeFeature.hits.col(j);
                Eigen::MatrixXf::Index closestHit;
                // TODO: Check this can't crash.
                auto hitDistance = (otherFeatures.hits.colwise() - hit).cwiseAbs().colwise().squaredNorm();
                hitDistance.minCoeff(&closestHit);

                if (hitDistance[closestHit] < closestApproach)
                    closestApproach = hitDistance[closestHit];
            }

            edges.push_back({(int)currentNode, indices[i].col});
            edgeFeatures.push_back({0.f, closestApproach / m_scalingFactor, centreDist / m_scalingFactor, angleBetween});
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DlShowerGrowingAlgorithm::BuildGraph(const Cluster *inputCluster, NodeFeatureVector &nodes, EdgeVector &edges,
    EdgeFeatureVector &edgeFeatures, LArDLHelper::TorchInputVector &inputs)
{
    LArDLHelper::TorchInput nodeTensor, edgeTensor, edgeAttrTensor;

    const int numNodes = nodes.size();
    const int numEdges = edges.size();

    static const int numNodeFeatures = 6;
    static const int edgeShape = 2;
    static const int numEdgeFeatures = 4;

    const auto asFloat = torch::TensorOptions().dtype(torch::kFloat32);
    const auto asInt = torch::TensorOptions().dtype(torch::kInt64);
    LArDLHelper::InitialiseInput({numNodes, numNodeFeatures}, nodeTensor, asFloat);
    LArDLHelper::InitialiseInput({edgeShape, numEdges}, edgeTensor, asInt);
    LArDLHelper::InitialiseInput({numEdges, numEdgeFeatures}, edgeAttrTensor, asFloat);

    int inputClusterNodeNum = 0;

    for (unsigned int i = 0; i < nodes.size(); i++)
    {
        NodeFeature info = nodes[i];
        const float isInput = info.cluster == inputCluster;

        if (isInput)
            ++inputClusterNodeNum;

        // INFO: We scale these larger values into more reasonable ranges.
        const float xMean = info.xMean / m_scalingFactor;
        const float zMean = info.zMean / m_scalingFactor;
        const float vtxDisp = info.vertexDisplacement / m_scalingFactor;

        std::vector<float> features = {isInput, info.numOfHits, info.orientation, xMean, zMean, vtxDisp};

        // ATTN: from_blob does not take ownership of the data!
        nodeTensor.slice(0, i, i + 1) = torch::from_blob(features.data(), {6}, asFloat);
    }

    for (unsigned int i = 0; i < edges.size(); i++)
    {
        edgeTensor[0][i] = edges[i][0];
        edgeTensor[1][i] = edges[i][1];
        edgeAttrTensor.slice(0, i, i + 1) = torch::from_blob(edgeFeatures[i].data(), {4}, asFloat);
    }

    inputs.insert(inputs.end(), {nodeTensor, edgeTensor, edgeAttrTensor});

    std::cout << "Nodes: " << nodeTensor.sizes() << ", " << nodeTensor.dtype() << std::endl;
    std::cout << "Input cluster node size: " << inputClusterNodeNum << std::endl;
    std::cout << "Edges: " << edgeTensor.sizes() << ", " << edgeTensor.dtype() << std::endl;
    std::cout << "EdgeAttrs: " << edgeAttrTensor.sizes() << ", " << edgeAttrTensor.dtype() << std::endl;

    if (inputClusterNodeNum == 0)
        std::cout << "###########################" << std::endl
                  << "Could not find " << (inputCluster) << std::endl
                  << "###########################" << std::endl;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlShowerGrowingAlgorithm::GrowClusters(
    const std::string &listName, const Cluster *inputCluster, IdClusterMap &nodeMap, LArDLHelper::TorchOutput &output, ClusterList &clusters)
{
    std::map<const Cluster *, int> joinResults;
    ClusterList remainingClusters;

    // INFO: max returns a tuple, 0 is the value, 1 is in indicies.
    //       So by just using the indicies, we know if 0/1 was picked.
    auto networkResult = std::get<1>(output.max(1));

    int nMerged = 0;

    // INFO: Store a count of how each node that makes up the cluster scored.
    //       A final positive score means more nodes were added than not.
    // TODO: This is just 1 > 0, no measure of how strong 1 is.
    for (unsigned int i = 0; i < nodeMap.size(); ++i)
    {
        const int currentResult = networkResult[i].item<int>();

        if (joinResults.count(nodeMap[i]) == 0)
            joinResults[nodeMap[i]] = 0;

        if (currentResult == 1)
            joinResults[nodeMap[i]] += 1;
        else
            joinResults[nodeMap[i]] -= 1;
    }

    std::cout << "The input cluster was of size " << inputCluster->GetNCaloHits() << " to start..." << std::endl;

    for (auto clusterResult : joinResults)
    {
        const auto cluster = clusterResult.first;
        const auto result = clusterResult.second;

        if (result > 0 && cluster != inputCluster)
        {
            PANDORA_RETURN_RESULT_IF(
                STATUS_CODE_SUCCESS, !=, PandoraContentApi::MergeAndDeleteClusters(*this, inputCluster, cluster, listName, listName));
            ++nMerged;
        }
        else if (cluster != inputCluster)
            remainingClusters.push_back(cluster);
    }

    std::cout << "There was " << nMerged << " clusters that were merged!" << std::endl;
    std::cout << "Input cluster was of size " << inputCluster->GetNCaloHits() << " after merging..." << std::endl;

    clusters.empty();
    clusters.assign(remainingClusters.begin(), remainingClusters.end());

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DlShowerGrowingAlgorithm::ProduceTrainingFile(const ClusterList *clusters, const std::string &clusterListName) const
{
    // No point training on something with a single cluster, nothing to learn.
    if (clusters->size() <= 1)
        return;

    const std::string fileName = m_trainingOutputFile + "_" + clusterListName + ".csv";

    LArMvaHelper::MvaFeatureVector eventFeatures;
    eventFeatures.push_back(static_cast<double>(clusters->size()));

    LArMCParticleHelper::CaloHitToMCMap eventLevelCaloHitToMCMap;
    LArMCParticleHelper::MCContributionMap eventLevelMCToCaloHitMap;
    std::map<const MCParticle *, int> mcIDMap; // Populated as its used.
    this->GetMCMaps(clusters, clusterListName, eventLevelCaloHitToMCMap, eventLevelMCToCaloHitMap);
    const Vertex *pVertex = nullptr;

    if (eventLevelCaloHitToMCMap.size() == 0 || eventLevelMCToCaloHitMap.size() == 0)
    {
        std::cout << "MC Map was empty..." << std::endl;
        return;
    }

    // Get the vertex "list" which seems to only be used for the first element, if at all.
    // Write that out first.
    try
    {
        const VertexList *pVertexList(nullptr);
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));

        if (pVertexList == nullptr)
            throw STATUS_CODE_NOT_FOUND;

        if (pVertexList->size() == 0)
            return;

        for (auto vertex : *pVertexList)
        {
            pVertex = vertex;
            const CartesianVector pos = vertex->GetPosition();
            eventFeatures.push_back(static_cast<double>(pos.GetX()));
            eventFeatures.push_back(static_cast<double>(pos.GetY()));
            eventFeatures.push_back(static_cast<double>(pos.GetZ()));
        }
    }
    catch (StatusCodeException)
    {
        return;
    }

    // Populate MC -> (int) ID map, so it will be full for writing.
    // TODO: Actually, this can be a little incomplete somehow? Dynamically
    // add these special cases when needed.
    eventFeatures.push_back(static_cast<double>(eventLevelMCToCaloHitMap.size()));

    for (auto &mcCaloListPair : eventLevelMCToCaloHitMap)
    {
        const auto mc = mcCaloListPair.first;

        eventFeatures.push_back(static_cast<double>(this->GetIdForMC(mc, mcIDMap)));
        eventFeatures.push_back(static_cast<double>(mc->GetParticleId()));
    }

    int clusterNumber = 0;

    for (auto const &cluster : *clusters)
    {

        // Get all calo hits for this cluster.

        CaloHitList clusterCaloHits;
        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList())
        {
            CaloHitList hitsForCluster(*clusterHitPair.second);
            clusterCaloHits.merge(hitsForCluster);
        }

        if (clusterCaloHits.size() == 0)
            continue;

        const MCParticle *pMCParticle = nullptr;

        try
        {
            pMCParticle = MCParticleHelper::GetMainMCParticle(cluster);
        }
        catch (const StatusCodeException &)
        {
            continue; // TODO: Should we skip these clusters? Harder to train on
        }

        const int cId = cluster->GetParticleId();

        LArMvaHelper::MvaFeatureVector clusterFeatures;
        LArMvaHelper::MvaFeatureVector hitFeatures;
        clusterFeatures.push_back(static_cast<double>(clusterNumber));
        clusterFeatures.push_back(static_cast<double>(cId));

        if (mcIDMap.count(pMCParticle) == 0)
        {
            std::cout << "Can't find a unique ID for this cluster, adding!" << std::endl;
            eventFeatures.push_back(static_cast<double>(this->GetIdForMC(pMCParticle, mcIDMap)));
            eventFeatures.push_back(static_cast<double>(pMCParticle->GetParticleId()));
            eventFeatures[4] = eventFeatures[4].Get() + 1.0; // Increment the mc count.
        }

        clusterFeatures.push_back(this->GetIdForMC(pMCParticle, mcIDMap));

        // TODO: These numbers are directly copied from the ShowerGrowing, change?
        const double direction((nullptr == pVertex) ? LArVertexHelper::DIRECTION_UNKNOWN
                                                    : LArVertexHelper::GetClusterDirectionInZ(this->GetPandora(), pVertex, cluster, 1.732f, 0.333f));
        clusterFeatures.push_back(direction);

        int hitNumber = 0;
        for (const auto caloHit : clusterCaloHits)
        {

            const CartesianVector pos = caloHit->GetPositionVector();
            const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);

            if (it2 == eventLevelCaloHitToMCMap.end())
                continue; // TODO: Failed MC, keep?

            const auto mc = it2->second;

            if (mcIDMap.count(mc) == 0)
            {
                std::cout << "Can't find a unique ID for this hit!" << std::endl;
                eventFeatures.push_back(static_cast<double>(this->GetIdForMC(mc, mcIDMap)));
                eventFeatures.push_back(static_cast<double>(mc->GetParticleId()));
                eventFeatures[4] = eventFeatures[4].Get() + 1.0; // Increment the mc count.
            }

            hitFeatures.push_back(static_cast<double>(pos.GetX()));
            hitFeatures.push_back(static_cast<double>(pos.GetZ()));
            hitFeatures.push_back(this->GetIdForMC(mc, mcIDMap));

            ++hitNumber;
        }

        if (hitFeatures.size() == 0)
            continue;

        clusterFeatures.push_back(static_cast<double>(hitFeatures.size() / 3.0));

        LArMvaHelper::ProduceTrainingExample(fileName, true, eventFeatures, clusterFeatures, hitFeatures);
        ++clusterNumber;
    }

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DlShowerGrowingAlgorithm::Visualize(const LArDLHelper::TorchInput nodeTensor, const LArDLHelper::TorchInput edgeTensor,
    const LArDLHelper::TorchOutput output, const std::string &listName) const
{
    using namespace torch::indexing;

    const std::string inputHitsName("InputHits_" + listName);
    const std::string selectedHitsName("SelectedHits_" + listName);
    const std::string backgroundHitsName("BackgroundHits_" + listName);
    CaloHitList inputHits, selectedHits, backgroundHits;

    for (int i = 0; i < nodeTensor.size(0); ++i)
    {
        const auto node = nodeTensor[i];
        const auto result = output[i];

        const CartesianVector hit({node[3].item<float>(), 0.f, node[4].item<float>()});

        if (node[0].item<float>() == 1.0)
        {

            PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &hit, inputHitsName, RED, 2));
        }
        else if (result[1].item<float>() > result[0].item<float>())
        {

            PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &hit, selectedHitsName, BLUE, 2));
        }
        else
        {

            PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &hit, backgroundHitsName, BLACK, 2));
        }
    }

    for (int i = 0; i < edgeTensor.size(1); ++i)
    {
        const auto startIdx = edgeTensor[0][i].item<int>();
        const auto endIdx = edgeTensor[1][i].item<int>();

        const auto startNode = nodeTensor[startIdx];
        const auto endNode = nodeTensor[endIdx];

        const CartesianVector startPos({startNode[3].item<float>(), 0.f, startNode[4].item<float>()});
        const CartesianVector endPos({endNode[3].item<float>(), 0.f, endNode[4].item<float>()});

        PANDORA_MONITORING_API(AddLineToVisualization(this->GetPandora(), &startPos, &endPos, "edge", GRAY, 2, 1));
    }

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DlShowerGrowingAlgorithm::GetMCMaps(const ClusterList * /*clusterList*/, const std::string &clusterListName,
    LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, LArMCParticleHelper::MCContributionMap &MCtoCaloMap) const
{
    const MCParticleList *pMCParticleList = nullptr;

    try
    {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "Input", pMCParticleList));
    }
    catch (StatusCodeException e)
    {
        std::cout << "Failed to get MCParticleList: " << e.ToString() << std::endl;
        return;
    }

    LArMCParticleHelper::MCRelationMap mcToTargetMCMap;
    LArMCParticleHelper::GetMCToSelfMap(pMCParticleList, mcToTargetMCMap);

    const CaloHitList *pCaloHitList = nullptr;
    std::string caloListName("CaloHitList");
    caloListName += clusterListName.back();

    try
    {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, caloListName, pCaloHitList));
    }
    catch (StatusCodeException e)
    {
        std::cout << "Failed to get CaloHitList: " << e.ToString() << std::endl;
        return;
    }

    CaloHitList caloHits(*pCaloHitList);

    //    // Subtract from the full calo hit list the "actually" reconstructible hits.
    //    // Otherwise, 100% is impossible as some hits are skipped when in tiny clusters.
    //    // This should give a fairer/more accurate completeness/purity.
    //    for (auto const &cluster : *clusterList) {
    //
    //        if (cluster->GetNCaloHits() >= 5)
    //            continue;
    //
    //        for (const auto &clusterHitPair : cluster->GetOrderedCaloHitList())
    //            for (const auto hit : *clusterHitPair.second)
    //                caloHits.remove(hit);
    //
    //        for (const auto hit : cluster->GetIsolatedCaloHitList())
    //            caloHits.remove(hit);
    //    }

    try
    {
        LArMCParticleHelper::GetMCParticleToCaloHitMatches(&(caloHits), mcToTargetMCMap, caloToMCMap, MCtoCaloMap);
    }
    catch (StatusCodeException e)
    {
        std::cout << "Failed to get matches: " << e.ToString() << std::endl;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

double DlShowerGrowingAlgorithm::GetIdForMC(const MCParticle *mc, std::map<const MCParticle *, int> &idMap) const
{
    if (idMap.count(mc) == 0)
        idMap[mc] = idMap.size();

    return idMap[mc];
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlShowerGrowingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "UseTrainingMode", m_useTrainingMode));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "RecoStatus", m_recoStatus));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "LimitedEdges", m_limitedEdges));

    if (m_useTrainingMode)
    {
        PANDORA_RETURN_RESULT_IF_AND_IF(
            STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "TrainingFileName", m_trainingOutputFile));
    }
    else
    {
        // TODO: Re-enable once needed!

        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileNameU", m_modelFileNameU));
        m_modelFileNameU = LArFileHelper::FindFileInPath(m_modelFileNameU, "FW_SEARCH_PATH");
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, LArDLHelper::LoadModel(m_modelFileNameU, m_modelU));
        m_modelU.eval();
        // PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileNameV", m_modelFileNameV));
        // m_modelFileNameV = LArFileHelper::FindFileInPath(m_modelFileNameV, "FW_SEARCH_PATH");
        // PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, LArDLHelper::LoadModel(m_modelFileNameV, m_modelV));
        // PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileNameW", m_modelFileNameW));
        // m_modelFileNameW = LArFileHelper::FindFileInPath(m_modelFileNameW, "FW_SEARCH_PATH");
        // PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, LArDLHelper::LoadModel(m_modelFileNameW, m_modelW));
    }

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadVectorOfValues(xmlHandle, "InputClusterListNames", m_clusterListNames));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "Visualize", m_visualize));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_dl_content
