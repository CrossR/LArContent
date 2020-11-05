/**
 *  @file   larpandoracontent/LArTrackShowerId/ShowerGrowingAlgorithm.cc
 *
 *  @brief  Implementation of the shower growing algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"

#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArPointingClusterHelper.h"

#include "larpandoracontent/LArObjects/LArPointingCluster.h"

#include "larpandoracontent/LArTrackShowerId/ShowerGrowingAlgorithm.h"

// TODO: Debugging, remove
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"
#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>

using namespace pandora;

namespace lar_content
{

ShowerGrowingAlgorithm::ShowerGrowingAlgorithm() :
    m_minCaloHitsPerCluster(5),
    m_nearbyClusterDistance(2.5f),
    m_remoteClusterDistance(10.f),
    m_directionTanAngle(1.732f),
    m_directionApexShift(0.333f),
    m_minVertexLongitudinalDistance(-2.5f),
    m_maxVertexLongitudinalDistance(20.f),
    m_maxVertexTransverseDistance(1.5f),
    m_vertexAngularAllowance(3.f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ShowerGrowingAlgorithm::IsVertexAssociated(const LArPointingCluster &pointingCluster, const CartesianVector &vertexPosition2D) const
{
    return (LArPointingClusterHelper::IsNode(vertexPosition2D, pointingCluster.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(vertexPosition2D, pointingCluster.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsEmission(vertexPosition2D, pointingCluster.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(vertexPosition2D, pointingCluster.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ShowerGrowingAlgorithm::SortClusters(const Cluster *const pLhs, const Cluster *const pRhs)
{
    CartesianVector innerCoordinateLhs(0.f, 0.f, 0.f), outerCoordinateLhs(0.f, 0.f, 0.f);
    LArClusterHelper::GetExtremalCoordinates(pLhs, innerCoordinateLhs, outerCoordinateLhs);
    const float dLhs2((outerCoordinateLhs - innerCoordinateLhs).GetMagnitudeSquared());

    CartesianVector innerCoordinateRhs(0.f, 0.f, 0.f), outerCoordinateRhs(0.f, 0.f, 0.f);
    LArClusterHelper::GetExtremalCoordinates(pRhs, innerCoordinateRhs, outerCoordinateRhs);
    const float dRhs2((outerCoordinateRhs - innerCoordinateRhs).GetMagnitudeSquared());

    return (dLhs2 > dRhs2);
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ShowerGrowingAlgorithm::Run()
{
    for (const std::string &clusterListName : m_inputClusterListNames)
    {
        try
        {
            const ClusterList *pClusterList = nullptr;
            PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, clusterListName, pClusterList));

            if (!pClusterList || pClusterList->empty())
            {
                if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
                    std::cout << "ShowerGrowingAlgorithm: unable to find cluster list " << clusterListName << std::endl;

                continue;
            }

            this->DumpClusterList(clusterListName, "Before");
            this->SimpleModeShowerGrowing(pClusterList, clusterListName);
            this->DumpClusterList(clusterListName, "After");
            m_clusterDirectionMap.clear();
        }
        catch (StatusCodeException &statusCodeException)
        {
            m_clusterDirectionMap.clear();
            throw statusCodeException;
        }
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerGrowingAlgorithm::DumpClusterList(const std::string &clusterListName, const std::string &recoStatus) const
{
    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    while (true)
    {
        fileName = "/home/scratch/showerClusters/clusters_" +
            clusterListName + "_" + recoStatus + "_" +
            std::to_string(fileNum);
        std::ifstream testFile(fileName + ".csv");

        if (!testFile.good())
            break;

        testFile.close();
        ++fileNum;
    }

    std::ofstream csvFile;
    csvFile.open(fileName + ".csv");

    const ClusterList *pClusterList = nullptr;

    try {
        PANDORA_THROW_RESULT_IF_AND_IF(
            STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=,
            PandoraContentApi::GetList(*this, clusterListName, pClusterList)
        );
    } catch (StatusCodeException e) {
        std::cout << "Failed to get cluster list: " << e.ToString() << std::endl;
        csvFile.close();
        return;
    }

    // Sanity checks: General cluster size should go up.
    //                Number of clusters should go down.
    //                Purity should remain same-ish
    //                Completeness should go up.

    const MCParticleList *pMCParticleList = nullptr;

    try {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "Input", pMCParticleList));
    } catch (StatusCodeException e) {
        std::cout << "Failed to get MCParticleList: " << e.ToString() << std::endl;
        csvFile.close();
        return;
    }

    // MC Setup.
    // Build up Calo Hit -> Map and MC -> CaloHitList.
    LArMCParticleHelper::MCRelationMap mcToTargetMCMap;
    LArMCParticleHelper::GetMCToSelfMap(pMCParticleList, mcToTargetMCMap);

    LArMCParticleHelper::CaloHitToMCMap eventLevelCaloHitToMCMap;
    LArMCParticleHelper::MCContributionMap eventLevelMCToCaloHitMap;

    // For each cluster, build up the two maps (CaloHit -> MC, MC -> CaloHitList).
    // To do this, we need to get all the calo hits from the cluster, then
    // call the helper.
    //
    // Once we've done that add the CalotHit -> MC bits, and merge the
    // MC -> CaloHitList stuff.
    //
    // This is because a hit is to 1 MC particle, whereas the MC to CaloHitList
    // is to a list, and that list is incomplete (only contains hits from the
    // current cluster).
    for (auto const &cluster : *pClusterList) {

        LArMCParticleHelper::CaloHitToMCMap perClusterCaloHitToMCMap;
        LArMCParticleHelper::MCContributionMap perClusterMCToCaloHitMap;

        CaloHitList caloHits;
        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            CaloHitList hitsForCluster(*clusterHitPair.second);
            caloHits.merge(hitsForCluster);
        }
        CaloHitList isolatedHits = cluster->GetIsolatedCaloHitList();
        caloHits.merge(isolatedHits);

        try {
            LArMCParticleHelper::GetMCParticleToCaloHitMatches(
                &(caloHits), mcToTargetMCMap, perClusterCaloHitToMCMap, perClusterMCToCaloHitMap
            );
        } catch (StatusCodeException e) {
            std::cout << "Failed to get matches: " << e.ToString() << std::endl;
        }

        eventLevelCaloHitToMCMap.insert(perClusterCaloHitToMCMap.begin(), perClusterCaloHitToMCMap.end());

        // Merge in the MC -> CaloHitList stuff if needed, otherwise just add all.
        for (auto &mcCaloHitListPair : perClusterMCToCaloHitMap) {
            const auto it = eventLevelMCToCaloHitMap.find(mcCaloHitListPair.first);

            if (it == eventLevelMCToCaloHitMap.end()) {
                eventLevelMCToCaloHitMap.insert(mcCaloHitListPair);
            } else {
                it->second.merge(mcCaloHitListPair.second);
                it->second.unique();
            }
        }
    }

    // ROOT TTree variable setup
    int clusterNumber = -1;
    float completeness = 0.0;
    float purity = 0.0;
    float failedHits = 0.0;
    float matchesMain = 0.0;
    float len = 0.0;
    int mcID = 0;

    const std::string treeName = "showerClustersTree";
    PANDORA_MONITORING_API(Create(this->GetPandora()));

    for (auto const &cluster : *pClusterList) {

        ++clusterNumber;
        completeness = 0.0;
        purity = 0.0;
        len = 0.0;
        matchesMain = 0.0;
        failedHits = 0.0;
        mcID = 0;

        csvFile << "X, Z, Type, PID, IsAvailable, IsShower, MCUid, IsIsolated" << std::endl;

        const MCParticle *pMCParticle = nullptr;
        Uid mcUid = nullptr;
        float hitsInMC = 0.0;

        try {
            pMCParticle = MCParticleHelper::GetMainMCParticle(cluster);
            mcUid = pMCParticle->GetUid();
            mcID = pMCParticle->GetParticleId();
        } catch (const StatusCodeException &) {
            // TODO: Attach debugger and check why!
            int c_size = cluster->GetOrderedCaloHitList().size() + cluster->GetIsolatedCaloHitList().size();
            std::cout << "  ## No MC. Size " << c_size << std::endl;
            mcUid = (const void*) -999;
            mcID = -999;
        }

        auto mcToCaloHit = eventLevelMCToCaloHitMap.find(pMCParticle);
        if (mcToCaloHit != eventLevelMCToCaloHitMap.end()) {
            hitsInMC = mcToCaloHit->second.size();
        } else {
            std::cout << " >> Failed to find MC in MC -> Calo map!" << std::endl;
            hitsInMC = -999;
        }

        const int isShower = std::abs(cluster->GetParticleId()) == MU_MINUS ? 0 : 1;

        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            for (auto const &caloHit : *(clusterHitPair.second)) {
                ++len;

                const CartesianVector pos = caloHit->GetPositionVector();
                csvFile << pos.GetX() << ", "
                        << pos.GetZ() << ", "
                        << recoStatus << ", "
                        << cluster->GetParticleId() << ", "
                        << cluster->IsAvailable() << ", "
                        << isShower << ", "
                        << mcUid << ", "
                        << "0" << std::endl;

                const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);

                if (it2 == eventLevelCaloHitToMCMap.end()) {
                   ++failedHits;
                   continue;
                }

                const auto mc = it2->second;

                if (mc == pMCParticle) {
                    ++matchesMain;
                }
            }
        }

        for (auto const &caloHit : cluster->GetIsolatedCaloHitList()) {
            ++len;

            const CartesianVector pos = caloHit->GetPositionVector();
            csvFile << pos.GetX() << ", "
                    << pos.GetZ() << ", "
                    << recoStatus << ", "
                    << cluster->GetParticleId() << ", "
                    << cluster->IsAvailable() << ", "
                    << isShower << ", "
                    << mcUid << ", "
                    << "1" << std::endl;

            const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);

            if (it2 == eventLevelCaloHitToMCMap.end()) {
               ++failedHits;
               continue;
            }

            const auto mc = it2->second;

            if (mc == pMCParticle) {
                ++matchesMain;
            }
        }

        if (cluster->GetIsolatedCaloHitList().size() != 0)
            std::cout << " #### Isolated hits in this event! ####" << std::endl;

        completeness = matchesMain / hitsInMC;
        purity = matchesMain / len;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "clusterNumber", clusterNumber));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "completeness", completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "purity", purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "numberOfHits", len));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "failedHits", failedHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "mcID", mcID));
        PANDORA_MONITORING_API(FillTree(this->GetPandora(), treeName));
    }

    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), treeName, fileName + ".root", "RECREATE"));
    // TODO: Maybe something higher level too?
    // Right now, we have "This cluster is X% complete and X% pure".
    // Flipping it to "This MC Particle is spread across X clusters, with X purity" could also be nice.
    PANDORA_MONITORING_API(Delete(this->GetPandora()));
    csvFile.close();
    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerGrowingAlgorithm::SimpleModeShowerGrowing(const ClusterList *const pClusterList, const std::string &clusterListName) const
{
    const VertexList *pVertexList(nullptr);
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));
    const Vertex *const pVertex(((pVertexList->size() == 1) && (VERTEX_3D == (*(pVertexList->begin()))->GetVertexType())) ? *(pVertexList->begin()) : nullptr);

    ClusterSet usedClusters;

    // Pick up all showers starting at vertex
    if (pVertex)
    {
        ClusterVector seedClusters;
        this->GetAllVertexSeedCandidates(pClusterList, pVertex, seedClusters);

        SeedAssociationList vertexSeedAssociationList;
        this->GetSeedAssociationList(seedClusters, pClusterList, vertexSeedAssociationList);
        this->ProcessSeedAssociationDetails(vertexSeedAssociationList, clusterListName, usedClusters);
    }

    // Non-vertex showers
    const Cluster *pSeedCluster(nullptr);

    while (this->GetNextSeedCandidate(pClusterList, usedClusters, pSeedCluster))
    {
        SeedAssociationList seedAssociationList;
        this->GetSeedAssociationList(ClusterVector(1, pSeedCluster), pClusterList, seedAssociationList);
        this->ProcessSeedAssociationDetails(seedAssociationList, clusterListName, usedClusters);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ShowerGrowingAlgorithm::GetNextSeedCandidate(const ClusterList *const pClusterList, const ClusterSet &usedClusters,
    const Cluster *&pSeedCluster) const
{
    pSeedCluster = nullptr;

    ClusterVector clusterVector;
    clusterVector.insert(clusterVector.end(), pClusterList->begin(), pClusterList->end());
    std::sort(clusterVector.begin(), clusterVector.end(), ShowerGrowingAlgorithm::SortClusters);

    for (const Cluster *const pCluster : clusterVector)
    {
        if (!pCluster->IsAvailable())
            continue;

        if (MU_MINUS == std::abs(pCluster->GetParticleId()))
            continue;

        if (pCluster->GetNCaloHits() < m_minCaloHitsPerCluster)
            continue;

        if (usedClusters.count(pCluster))
            continue;

        pSeedCluster = pCluster;
        return true;
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerGrowingAlgorithm::GetAllVertexSeedCandidates(const ClusterList *const pClusterList, const Vertex *const pVertex,
    ClusterVector &seedClusters) const
{
    ClusterVector clusterVector;
    clusterVector.insert(clusterVector.end(), pClusterList->begin(), pClusterList->end());

    if (clusterVector.empty())
        return;

    const HitType hitType(LArClusterHelper::GetClusterHitType(clusterVector.at(0)));
    const CartesianVector vertexPosition2D(LArGeometryHelper::ProjectPosition(this->GetPandora(), pVertex->GetPosition(), hitType));

    for (const Cluster *const pCluster : clusterVector)
    {
        if (!pCluster->IsAvailable())
            continue;

        if (MU_MINUS == std::abs(pCluster->GetParticleId()))
            continue;

        if (pCluster->GetNCaloHits() < m_minCaloHitsPerCluster)
            continue;

        try
        {
            if (this->IsVertexAssociated(LArPointingCluster(pCluster), vertexPosition2D))
                seedClusters.push_back(pCluster);
        }
        catch (StatusCodeException &)
        {
        }
    }

    std::sort(seedClusters.begin(), seedClusters.end(), ShowerGrowingAlgorithm::SortClusters);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerGrowingAlgorithm::GetSeedAssociationList(const ClusterVector &particleSeedVector, const ClusterList *const pClusterList,
    SeedAssociationList &seedAssociationList) const
{
    if (particleSeedVector.empty())
        return;

    ClusterVector candidateClusters;
    const ClusterList clusterList(*pClusterList);

    for (const Cluster *const pCandidateCluster : clusterList)
    {
        if (!pCandidateCluster->IsAvailable())
            continue;

        if (MU_MINUS == std::abs(pCandidateCluster->GetParticleId()))
            continue;

        if (pCandidateCluster->GetNCaloHits() < m_minCaloHitsPerCluster)
            continue;

        if (particleSeedVector.end() == std::find(particleSeedVector.begin(), particleSeedVector.end(), pCandidateCluster))
            candidateClusters.push_back(pCandidateCluster);
    }

    std::sort(candidateClusters.begin(), candidateClusters.end(), ShowerGrowingAlgorithm::SortClusters);
    ClusterUsageMap forwardUsageMap, backwardUsageMap;

    for (const Cluster *const pSeedCluster : particleSeedVector)
    {
        this->FindAssociatedClusters(pSeedCluster, candidateClusters, forwardUsageMap, backwardUsageMap);
    }

    this->IdentifyClusterMerges(particleSeedVector, backwardUsageMap, seedAssociationList);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerGrowingAlgorithm::ProcessSeedAssociationDetails(const SeedAssociationList &seedAssociationList, const std::string &clusterListName,
    ClusterSet &usedClusters) const
{
    ClusterList clusterList;
    for (const auto &mapEntry : seedAssociationList) clusterList.push_back(mapEntry.first);
    clusterList.sort(LArClusterHelper::SortByNHits);

    for (const Cluster *const pParentCluster : clusterList)
    {
        const ClusterVector &branchClusters(seedAssociationList.at(pParentCluster));
        this->ProcessBranchClusters(pParentCluster, branchClusters, clusterListName);

        usedClusters.insert(pParentCluster);
        usedClusters.insert(branchClusters.begin(), branchClusters.end());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerGrowingAlgorithm::ProcessBranchClusters(const Cluster *const pParentCluster, const ClusterVector &branchClusters, const std::string &listName) const
{
    m_clusterDirectionMap.erase(pParentCluster);

    for (const Cluster *const pBranchCluster : branchClusters)
    {
        if (pBranchCluster->IsAvailable())
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::MergeAndDeleteClusters(*this, pParentCluster, pBranchCluster, listName, listName));
        }

        m_clusterDirectionMap.erase(pBranchCluster);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

ShowerGrowingAlgorithm::AssociationType ShowerGrowingAlgorithm::AreClustersAssociated(const Cluster *const pClusterSeed, const Cluster *const pCluster) const
{
    const VertexList *pVertexList(nullptr);
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));
    const Vertex *const pVertex(((pVertexList->size() == 1) && (VERTEX_3D == (*(pVertexList->begin()))->GetVertexType())) ? *(pVertexList->begin()) : nullptr);

    // Direction of seed cluster (cache for efficiency)
    ClusterDirectionMap::const_iterator seedIter = m_clusterDirectionMap.find(pClusterSeed);

    if (m_clusterDirectionMap.end() == seedIter)
    {
        const LArVertexHelper::ClusterDirection direction((nullptr == pVertex) ? LArVertexHelper::DIRECTION_UNKNOWN :
            LArVertexHelper::GetClusterDirectionInZ(this->GetPandora(), pVertex, pClusterSeed, m_directionTanAngle, m_directionApexShift));
        seedIter = m_clusterDirectionMap.insert(ClusterDirectionMap::value_type(pClusterSeed, direction)).first;
    }

    const LArVertexHelper::ClusterDirection seedDirection(seedIter->second);
    const bool checkSeedForward(seedDirection != LArVertexHelper::DIRECTION_BACKWARD_IN_Z);
    const bool checkSeedBackward(seedDirection != LArVertexHelper::DIRECTION_FORWARD_IN_Z);

    // Direction of candidate cluster (cache for efficiency)
    ClusterDirectionMap::const_iterator candIter = m_clusterDirectionMap.find(pCluster);

    if (m_clusterDirectionMap.end() == candIter)
    {
        const LArVertexHelper::ClusterDirection direction((nullptr == pVertex) ? LArVertexHelper::DIRECTION_UNKNOWN :
            LArVertexHelper::GetClusterDirectionInZ(this->GetPandora(), pVertex, pCluster, m_directionTanAngle, m_directionApexShift));
        candIter = m_clusterDirectionMap.insert(ClusterDirectionMap::value_type(pCluster, direction)).first;
    }

    const LArVertexHelper::ClusterDirection candidateDirection(candIter->second);
    const bool checkCandidateForward(candidateDirection != LArVertexHelper::DIRECTION_BACKWARD_IN_Z);
    const bool checkCandidateBackward(candidateDirection != LArVertexHelper::DIRECTION_FORWARD_IN_Z);

    // Calculate distances of association
    const float sOuter(LArClusterHelper::GetClosestDistance(pClusterSeed->GetCentroid(pClusterSeed->GetOuterPseudoLayer()), pCluster));
    const float cOuter(LArClusterHelper::GetClosestDistance(pCluster->GetCentroid(pCluster->GetOuterPseudoLayer()), pClusterSeed));
    const float sInner(LArClusterHelper::GetClosestDistance(pClusterSeed->GetCentroid(pClusterSeed->GetInnerPseudoLayer()), pCluster));
    const float cInner(LArClusterHelper::GetClosestDistance(pCluster->GetCentroid(pCluster->GetInnerPseudoLayer()), pClusterSeed));

    // Association check 1(a), look for enclosed clusters
    if ((cOuter < m_nearbyClusterDistance && cInner < m_nearbyClusterDistance) &&
        (!checkSeedForward || (sInner > m_nearbyClusterDistance)) &&
        (!checkSeedBackward || (sOuter > m_nearbyClusterDistance)))
    {
        return STRONG;
    }

    // Association check 1(b), look for overlapping clusters
    if ((checkSeedForward == checkCandidateForward) && (checkSeedBackward == checkCandidateBackward))
    {
        if ((cInner < m_nearbyClusterDistance && sOuter < m_nearbyClusterDistance) &&
            (!checkSeedForward || (sInner > m_nearbyClusterDistance)) &&
            (!checkSeedBackward || (cOuter > m_nearbyClusterDistance)))
        {
            return STRONG;
        }

        if ((cOuter < m_nearbyClusterDistance && sInner < m_nearbyClusterDistance) &&
            (!checkSeedBackward || (sOuter > m_nearbyClusterDistance)) &&
            (!checkSeedForward || (cInner > m_nearbyClusterDistance)))
        {
            return STRONG;
        }
    }

    // Association check 2, look for branching clusters
    if ((!checkSeedForward || (sInner > m_remoteClusterDistance)) &&
        (!checkSeedBackward || (sOuter > m_remoteClusterDistance)) &&
        ((checkCandidateForward && (cInner < m_nearbyClusterDistance)) || (checkCandidateBackward && (cOuter < m_nearbyClusterDistance))))
    {
        return STANDARD;
    }

    // Association check 3, look any distance below threshold
    if ((sOuter < m_nearbyClusterDistance) || (cOuter < m_nearbyClusterDistance) || (sInner < m_nearbyClusterDistance) || (cInner < m_nearbyClusterDistance))
        return SINGLE_ORDER;

    return NONE;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float ShowerGrowingAlgorithm::GetFigureOfMerit(const SeedAssociationList &seedAssociationList) const
{
    const VertexList *pVertexList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));
    const Vertex *const pVertex(((pVertexList->size() == 1) && (VERTEX_3D == (*(pVertexList->begin()))->GetVertexType())) ? *(pVertexList->begin()) : nullptr);

    // ATTN Consistently returning same value will accept all candidate cluster merges
    if (!pVertex)
        return -1.f;

    unsigned int nVertexAssociatedSeeds(0), nVertexAssociatedNonSeeds(0);

    ClusterList clusterList;
    for (const auto &mapEntry : seedAssociationList) clusterList.push_back(mapEntry.first);
    clusterList.sort(LArClusterHelper::SortByNHits);

    for (const Cluster *const pSeedCluster : clusterList)
    {
        const ClusterVector &associatedClusters(seedAssociationList.at(pSeedCluster));
        const HitType hitType(LArClusterHelper::GetClusterHitType(pSeedCluster));
        const CartesianVector vertex2D(LArGeometryHelper::ProjectPosition(this->GetPandora(), pVertex->GetPosition(), hitType));

        LArPointingClusterList pointingClusterSeedList;
        try {pointingClusterSeedList.push_back(LArPointingCluster(pSeedCluster));} catch (StatusCodeException &) {}

        LArPointingClusterList pointingClusterNonSeedList;
        for (const Cluster *const pAssociatedCluster : associatedClusters)
        {
            try {pointingClusterNonSeedList.push_back(LArPointingCluster(pAssociatedCluster));} catch (StatusCodeException &) {}
        }

        nVertexAssociatedSeeds += this->GetNVertexConnections(vertex2D, pointingClusterSeedList);
        nVertexAssociatedNonSeeds += this->GetNVertexConnections(vertex2D, pointingClusterNonSeedList);
    }

    const float figureOfMerit(static_cast<float>(nVertexAssociatedSeeds) - static_cast<float>(nVertexAssociatedNonSeeds));
    return figureOfMerit;
}

//------------------------------------------------------------------------------------------------------------------------------------------

unsigned int ShowerGrowingAlgorithm::GetNVertexConnections(const CartesianVector &vertexPosition2D, const LArPointingClusterList &pointingClusterList) const
{
    unsigned int nConnections(0);

    for (LArPointingClusterList::const_iterator cIter = pointingClusterList.begin(), cIterEnd = pointingClusterList.end(); cIter != cIterEnd; ++cIter)
    {
        if (this->IsVertexAssociated(*cIter, vertexPosition2D))
            ++nConnections;
    }

    return nConnections;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ShowerGrowingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "InputClusterListNames", m_inputClusterListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinCaloHitsPerCluster", m_minCaloHitsPerCluster));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "NearbyClusterDistance", m_nearbyClusterDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "RemoteClusterDistance", m_remoteClusterDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "DirectionTanAngle", m_directionTanAngle));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "DirectionApexShift", m_directionApexShift));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinVertexLongitudinalDistance", m_minVertexLongitudinalDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxVertexLongitudinalDistance", m_maxVertexLongitudinalDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxVertexTransverseDistance", m_maxVertexTransverseDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "VertexAngularAllowance", m_vertexAngularAllowance));

    return BranchGrowingAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_content
