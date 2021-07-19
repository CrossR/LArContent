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

DlShowerGrowingAlgorithm::DlShowerGrowingAlgorithm() : m_visualize(false), m_useTrainingMode(false), m_trainingOutputFile("")
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

        if (m_dumpClusterList)
            this->DumpClusterList(pClusterList, listName);
        else
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
        this->InferForView(pClusterList);
        auto t2 = std::chrono::high_resolution_clock::now();

        /* Getting number of milliseconds as an integer. */
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "It took " << ms_int.count() << " milliseconds to run for input " << listName << std::endl;
    }

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

StatusCode DlShowerGrowingAlgorithm::InferForView(const ClusterList *clusters)
{
    const VertexList *pVertexList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));

    if (pVertexList == nullptr)
        return STATUS_CODE_NOT_FOUND;

    if (pVertexList->size() == 0)
        return STATUS_CODE_NOT_FOUND;

    // TODO: Check! Is there more than 1 vertex?
    const Vertex *pVertex = nullptr;
    for (auto vertex : *pVertexList)
        pVertex = vertex;

    if (pVertex == nullptr)
        return STATUS_CODE_NOT_FOUND;

    const int multiple = 2;

    int clusterId = 0;

    std::vector<NodeFeature> nodeFeatures;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<float>> edgeFeatures;

    // For every cluster, round all the hits to the nearest 2.
    // Then, build up the required node features and store them.
    for (auto cluster : *clusters)
    {

        if (cluster->GetParticleId() == 13)
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
                float x = caloHit->GetPositionVector().GetX();
                float z = caloHit->GetPositionVector().GetZ();
                int roundedX = (x / multiple) * multiple;
                int roundedZ = (z / multiple) * multiple;
                std::pair<int, int> roundedPos = {roundedX, roundedZ};

                // INFO: If this rounded hit lies with another rounded hit, store them together.
                if (roundedClusters.count(roundedPos) == 0)
                {
                    // TODO: Is a no vertex case valid? If it is, update feature gen below.
                    // TODO: Suitable init value for matrix?
                    const float orientation(LArVertexHelper::GetClusterDirectionInZ(this->GetPandora(), pVertex, cluster, 1.732f, 0.333f));
                    Eigen::MatrixXf hits(2, 10);
                    hits.col(0) << x, z;
                    roundedClusters.insert({roundedPos, {hits, 1, x, z, orientation}});
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

        // Build sliding fit to get out a direction for the cluster.
        // Crucially, this is a per-cluster feature! Each of the sub-nodes
        // that a cluster is split in are likely too small to have a reasonable
        // definition of direction.
        const float slidingFitPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
        const int slidingFitWindow = 100;
        CartesianVector direction(0.f, 0.f, 0.f);

        // TODO: Check this! Its cutting off 1 hit "clusters".
        if (allCaloHitsForCluster.size() > 2)
        {
            TwoDSlidingFitResult fit(&allCaloHitsForCluster, slidingFitWindow, slidingFitPitch);
            direction = fit.GetAxisDirection();
        }
        else if (allCaloHitsForCluster.size() == 2)
            direction = allCaloHitsForCluster[1] - allCaloHitsForCluster[0];
        else
            continue;

        // Turn the rounded node into an actual feature vector.
        for (auto node : roundedClusters)
        {
            RoundedClusterInfo &info = node.second;

            // Due to resizing, the matrix may be too large, so resize to real size.
            info.hits.conservativeResize(Eigen::NoChange, info.numOfHits);

            float numHits = info.numOfHits;
            float xMean = info.totalX / numHits;
            float zMean = info.totalZ / numHits;
            float vertexDisplacement = (xMean - pVertex->GetPosition().GetX()) + (zMean - pVertex->GetPosition().GetZ());
            float isSelected = 0.f;

            // Store the node features
            // TODO: Somewhere needs to set the input feature!
            NodeFeature features = {clusterId, info.hits, direction, isSelected, numHits, info.orientation, xMean, zMean, vertexDisplacement};
            nodeFeatures.push_back(features);
        }

        clusterId += 1;
    }

    // Build up a (2, N) matrix of positions.
    // This matrix can be used to find the 5NN to each node.
    Eigen::MatrixXf allNodePositions(2, nodeFeatures.size());
    int nodeNum = 0;

    for (auto nodeFeature : nodeFeatures)
    {
        allNodePositions.col(nodeNum) << nodeFeature.xMean, nodeFeature.zMean;
        nodeNum += 1;
    }

    for (unsigned int currentNode = 0; currentNode < nodeFeatures.size(); ++currentNode)
    {

        auto nodeFeature = nodeFeatures[currentNode];
        int currentId = nodeFeature.clusterId;
        Eigen::VectorXf meanPos(2);
        meanPos << nodeFeature.xMean, nodeFeature.zMean;

        std::vector<MatrixIndex> indices(6, {-1, -1});
        std::vector<double> values(6, std::numeric_limits<double>::max());

        visit_lambda((allNodePositions.colwise() - meanPos).colwise().squaredNorm(),
            [&](double v, int row, int col)
            {
                if (nodeFeatures[col].clusterId != currentId && v < values[0])
                {
                    auto it = std::lower_bound(values.rbegin(), values.rend(), v);
                    int index = std::distance(begin(values), it.base()) - 1;

                    values[index] = v;
                    indices[index] = {row, col};
                }
                else if (nodeFeatures[col].clusterId == currentId)
                {
                    edges.push_back({(int)currentNode, col});
                    edgeFeatures.push_back({1.f, 0.f, 0.f, 0.f});
                }
            });

        for (unsigned int otherNode = 0; otherNode < indices.size(); ++otherNode)
        {
            auto otherFeatures = nodeFeatures[otherNode];
            float closestApproach = std::numeric_limits<double>::max();

            for (unsigned int i = 0; i < nodeFeature.hits.cols(); ++i)
            {
                auto hit = nodeFeature.hits.col(i);
                Eigen::MatrixXf::Index closestHit;
                // TODO: Check this can't crash.
                auto hitDistance = (otherFeatures.hits.colwise() - hit).colwise().squaredNorm();
                hitDistance.minCoeff(&closestHit);

                if (hitDistance[closestHit] < closestApproach)
                    closestApproach = hitDistance[closestHit];
            }

            float angle = nodeFeature.direction.GetOpeningAngle(otherFeatures.direction);

            float centerDist = values[otherNode];
            edges.push_back({(int)currentNode, indices[otherNode].col});
            edgeFeatures.push_back({0.f, closestApproach, centerDist, angle});
        }
    }

    std::cout << "Test built " << nodeFeatures.size() << " nodes!" << std::endl;
    std::cout << "Test built " << edges.size() << " edges!" << std::endl;

    LArDLHelper::TorchInput nodeTensor;
    LArDLHelper::TorchInput edgeTensor;
    LArDLHelper::TorchInput edgeAttrTensor;

    int numNodes = nodeFeatures.size();
    int numEdges = edges.size();

    static const int numNodeFeatures = 6;
    static const int edgeShape = 2;
    static const int numEdgeFeatures = 4;

    // TODO: Remove magic numbers!
    auto asFloat = torch::TensorOptions().dtype(torch::kFloat32);
    auto asInt = torch::TensorOptions().dtype(torch::kInt64);
    LArDLHelper::InitialiseInput({numNodes, numNodeFeatures}, nodeTensor, asFloat);
    LArDLHelper::InitialiseInput({edgeShape, numEdges}, edgeTensor, asInt);
    LArDLHelper::InitialiseInput({numEdges, numEdgeFeatures}, edgeAttrTensor, asFloat);

    for (unsigned int i = 0; i < nodeFeatures.size(); i++)
    {
        NodeFeature info = nodeFeatures[i];
        std::vector<float> features = {info.isInputCluster, info.numOfHits, info.orientation, info.xMean, info.zMean, info.vertexDisplacement};

        // ATTN: from_blob does not take ownership of the data!
        nodeTensor.slice(0, i, i + 1) = torch::from_blob(features.data(), {6}, asFloat);
    }

    for (unsigned int i = 0; i < edges.size(); i++)
    {
        edgeTensor[0][i] = edges[i][0];
        edgeTensor[1][i] = edges[i][1];
        edgeAttrTensor.slice(0, i, i + 1) = torch::from_blob(edgeFeatures[i].data(), {4}, asFloat);
    }

    LArDLHelper::TorchInputVector inputs{nodeTensor, edgeTensor, edgeAttrTensor};
    LArDLHelper::TorchOutput output;
    LArDLHelper::TorchModel &model{m_modelU};

    std::cout << "Starting model!" << std::endl;
    std::cout << "Nodes: " << nodeTensor.sizes() << ", " << nodeTensor.dtype() << std::endl;
    std::cout << "Edges: " << edgeTensor.sizes() << ", " << edgeTensor.dtype() << std::endl;
    std::cout << "EdgeAttrs: " << edgeAttrTensor.sizes() << ", " << edgeAttrTensor.dtype() << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    LArDLHelper::Forward(model, inputs, output);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "It took " << ms_int.count() << " milliseconds to run inference" << std::endl;

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
        auto mc = mcCaloListPair.first;

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

void DlShowerGrowingAlgorithm::DumpClusterList(const ClusterList *clusters, const std::string &clusterListName) const
{
    // Pick folder.
    const std::string data_folder = "/home/lar/rcross/git/data_dir/showerClusters";
    system(("mkdir -p " + data_folder).c_str());

    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    while (true)
    {
        fileName = data_folder + "/clusters_" + clusterListName + "_" + m_recoStatus + "_" + std::to_string(fileNum);
        std::ifstream testFile(fileName + ".csv");

        if (!testFile.good())
            break;

        testFile.close();
        ++fileNum;
    }

    std::ofstream csvFile;
    csvFile.open(fileName + ".csv");

    LArMCParticleHelper::CaloHitToMCMap eventLevelCaloHitToMCMap;
    LArMCParticleHelper::MCContributionMap eventLevelMCToCaloHitMap;
    this->GetMCMaps(clusters, clusterListName, eventLevelCaloHitToMCMap, eventLevelMCToCaloHitMap);

    if (eventLevelCaloHitToMCMap.size() == 0 || eventLevelMCToCaloHitMap.size() == 0)
    {
        std::cout << "One of the MC Maps was empty..." << std::endl;
        csvFile.close();
        return;
    }

    const std::string treeName = "showerClustersTree";
    PANDORA_MONITORING_API(Create(this->GetPandora()));

    // Build up a map of MC -> Cluster ID, for the largest cluster.
    std::map<const MCParticle *, const Cluster *> mcToLargestClusterMap;
    std::map<const MCParticle *, int> mcIDMap; // Populated as its used.
    double largestShower = 0.0;

    for (auto const &cluster : *clusters)
    {
        try
        {
            if (std::abs(cluster->GetParticleId()) != MU_MINUS && cluster->GetNCaloHits() > largestShower)
                largestShower = cluster->GetNCaloHits();

            const MCParticle *mc = MCParticleHelper::GetMainMCParticle(cluster);

            if (mcToLargestClusterMap.count(mc) == 0)
            {
                mcToLargestClusterMap[mc] = cluster;
                continue;
            }

            const Cluster *currentCluster = mcToLargestClusterMap[mc];

            if (cluster->GetNCaloHits() > currentCluster->GetNCaloHits())
                mcToLargestClusterMap[mc] = cluster;
        }
        catch (const StatusCodeException &)
        {
            continue;
        }
    }

    int nFailed = 0;
    int nPassed = 0;

    for (auto const &cluster : *clusters)
    {

        // ROOT TTree variable setup
        double failedHits = 0.0;
        double matchesMain = 0.0;
        double clusterMainMCId = -999.0;
        double isLargestForMC = 0.0;

        // Get all calo hits for this cluster.
        CaloHitList clusterCaloHits;
        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList())
        {
            CaloHitList hitsForCluster(*clusterHitPair.second);
            clusterCaloHits.merge(hitsForCluster);
        }
        CaloHitList isolatedHits = cluster->GetIsolatedCaloHitList();
        clusterCaloHits.merge(isolatedHits);

        const MCParticle *pMCParticle = nullptr;
        double hitsInMC = -999.0;

        try
        {
            pMCParticle = MCParticleHelper::GetMainMCParticle(cluster);
            clusterMainMCId = pMCParticle->GetParticleId();

            if (mcToLargestClusterMap.count(pMCParticle) != 0)
                isLargestForMC = mcToLargestClusterMap[pMCParticle] == cluster ? 1.0 : 0.0;
        }
        catch (const StatusCodeException &)
        {
            // TODO: Attach debugger and check why!
            int c_size = cluster->GetOrderedCaloHitList().size() + cluster->GetIsolatedCaloHitList().size();
            std::cout << "  ## No MC. Size " << c_size << std::endl;
        }

        auto mcToCaloHit = eventLevelMCToCaloHitMap.find(pMCParticle);
        if (mcToCaloHit != eventLevelMCToCaloHitMap.end())
        {
            hitsInMC = mcToCaloHit->second.size();
            ++nPassed;
        }
        else
        {
            ++nFailed;
        }

        const int cId = cluster->GetParticleId();
        const int isShower = std::abs(cId) == MU_MINUS ? 0 : 1;
        const double isLargestShower = (isShower && cluster->GetNCaloHits() == largestShower) ? 1.0 : 0.0;

        // Write out the CSV file whilst building up info for the ROOT TTree.
        csvFile << "X, Z, Type, PID, IsAvailable, IsShower, MCId, IsIsolated, isVertex" << std::endl;

        // Get the vertex "list" which seems to only be used for the first element, if at all.
        // Write that out first.
        try
        {
            const VertexList *pVertexList(nullptr);
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));

            if (pVertexList == nullptr)
                throw STATUS_CODE_NOT_FOUND;

            for (const auto vertex : *pVertexList)
            {
                const CartesianVector pos = vertex->GetPosition();

                csvFile << pos.GetX() << ", " << pos.GetZ() << ", " << m_recoStatus << ", " << cluster->GetParticleId() << ", "
                        << cluster->IsAvailable() << ", 0, -999, 0, 1" << std::endl;
            }
        }
        catch (StatusCodeException)
        {
        }

        unsigned int index = 0;
        for (const auto caloHit : clusterCaloHits)
        {

            const CartesianVector pos = caloHit->GetPositionVector();
            const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);
            const bool isIsolated = index >= (clusterCaloHits.size() - cluster->GetIsolatedCaloHitList().size());
            int hitMCId = -999;

            if (it2 == eventLevelCaloHitToMCMap.end())
            {
                ++failedHits;
            }
            else
            {
                const auto mc = it2->second;
                hitMCId = this->GetIdForMC(mc, mcIDMap);

                if (mc == pMCParticle)
                {
                    ++matchesMain;
                }
            }

            csvFile << pos.GetX() << ", " << pos.GetZ() << ", " << m_recoStatus << ", " << cluster->GetParticleId() << ", "
                    << cluster->IsAvailable() << ", " << isShower << ", " << hitMCId << ", " << isIsolated << ", "
                    << "0" << std::endl;
            ++index;
        }

        // Finally, calculaate the completeness and purity, and write out TTree.
        const double completeness = matchesMain / hitsInMC;
        const double purity = matchesMain / clusterCaloHits.size();

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "clusterNumber", (double)clusters->size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "completeness", completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "purity", purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "numberOfHits", (double)clusterCaloHits.size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "failedHits", failedHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "mcID", clusterMainMCId));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "isShower", (double)isShower));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "tsIDCorrect", this->IsTaggedCorrectly(cId, clusterMainMCId)));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "isLargestForMC", isLargestForMC));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "isLargestShower", isLargestShower));
        PANDORA_MONITORING_API(FillTree(this->GetPandora(), treeName));
    }

    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), treeName, fileName + ".root", "RECREATE"));
    // TODO: Maybe something higher level too?
    // Right now, we have "This cluster is X% complete and X% pure".
    // Flipping it to "This MC Particle is spread across X clusters, with X purity" could also be nice.
    PANDORA_MONITORING_API(Delete(this->GetPandora()));

    std::cout << " >> Failed to find MC in MC -> Calo map for " << nFailed << " / " << nPassed + nFailed << std::endl;
    csvFile.close();

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

double DlShowerGrowingAlgorithm::IsTaggedCorrectly(const int cId, const int mcId) const
{

    std::vector<int> target;
    std::vector<int> showerLikeParticles({11, 22});
    std::vector<int> trackLikeParticles({13, 211, 2212, 321, 3222});

    if (std::abs(cId) == 11)
        target = showerLikeParticles;
    else
        target = trackLikeParticles;

    const auto it = std::find(target.begin(), target.end(), std::abs(mcId));

    return it != target.end();
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlShowerGrowingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "UseTrainingMode", m_useTrainingMode));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "DumpClusters", m_dumpClusterList));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "RecoStatus", m_recoStatus));

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
