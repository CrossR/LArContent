/**
 *  @file   larpandoracontent/LArUtility/ClusterDumpingAlgorithm.cc
 *
 *  @brief  Implementation for the debug cluster dumping algorithm class.
 *
 *  $Log: $
 */

#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"
#include "Objects/ParticleFlowObject.h"
#include "Pandora/AlgorithmHeaders.h"

#include "Pandora/PandoraEnumeratedTypes.h"
#include "Pandora/PandoraInternal.h"
#include "Pandora/StatusCodes.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArUtility/ClusterDumpingAlgorithm.h"

#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArMvaHelper.h"
#include "larpandoracontent/LArHelpers/LArVertexHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"

#include "larpandoracontent/LArObjects/LArTwoDSlidingFitResult.h"

#include <chrono>
#include <fstream>

#include <Eigen/Dense>

using namespace pandora;

namespace lar_content
{

StatusCode ClusterDumpingAlgorithm::Run()
{
    for (const std::string &view : m_viewNames)
    {

        ClusterList pClusterList({});
        const ClusterList *pTrackClusterList = nullptr;
        const ClusterList *pShowerClusterList = nullptr;

        try
        {
            if (m_nonSplit && view.length() == 1) {
                const std::string allClustersListName = "Clusters" + view;
                PANDORA_THROW_RESULT_IF_AND_IF(
                        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, allClustersListName, pTrackClusterList));
            } else if (m_nonSplit) {
                PANDORA_THROW_RESULT_IF_AND_IF(
                        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, view, pTrackClusterList));
            } else {
                const std::string trackListName = "TrackClusters" + view;
                PANDORA_THROW_RESULT_IF_AND_IF(
                        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, trackListName, pTrackClusterList));

                const std::string showerListName = "ShowerClusters" + view;
                PANDORA_THROW_RESULT_IF_AND_IF(
                        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, showerListName, pShowerClusterList));
            }
        }
        catch (StatusCodeException e)
        {
            std::cout << "Failed to get cluster list: " << e.ToString() << std::endl;
            continue;
        }

        if (pTrackClusterList != nullptr && pTrackClusterList->size() > 0)
            pClusterList.insert(pClusterList.end(), pTrackClusterList->begin(), pTrackClusterList->end());

        if (pShowerClusterList != nullptr && pShowerClusterList->size() > 0)
            pClusterList.insert(pClusterList.end(), pShowerClusterList->begin(), pShowerClusterList->end());

        if (pClusterList.size() == 0)
        {
            std::cout << "Cluster list was empty." << std::endl;
            continue;
        }

        if (m_dumpClusterList != "")
            this->DumpClusterInfo(&pClusterList, view);
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::DumpClusterInfo(const ClusterList *clusters, const std::string &clusterListName) const
{
    // Pick folder.
    const std::string data_folder = m_dumpClusterList;
    system(("mkdir -p " + data_folder).c_str());

    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    while (true)
    {
        fileName = data_folder + "/clusters_" + clusterListName + "_" + m_recoStatus + "_" + std::to_string(fileNum);
        std::ifstream testFile(fileName + ".root");

        if (!testFile.good())
            break;

        testFile.close();
        ++fileNum;
    }

    std::cout << "File name: " << fileName << std::endl;

    // Before any MC-based metrics, do the reco ones that can 100% be done.
    this->DumpRecoInfo(clusters, fileName);

    LArMCParticleHelper::CaloHitToMCMap eventLevelCaloHitToMCMap;
    LArMCParticleHelper::MCContributionMap eventLevelMCToCaloHitMap;
    const MCParticleList *pMCParticleList(nullptr);
    this->GetMCMaps(clusters, clusterListName, eventLevelCaloHitToMCMap, eventLevelMCToCaloHitMap, pMCParticleList);

    if (eventLevelCaloHitToMCMap.size() == 0 || eventLevelMCToCaloHitMap.size() == 0)
    {
        std::cout << "One of the MC Maps was empty..." << std::endl;
        return;
    }

    const std::string clusterTree = "showerClustersTree";
    const std::string mcTree = "mcInfoTree";
    PANDORA_MONITORING_API(Create(this->GetPandora()));

    // Build up a map of MC -> Cluster ID, for the largest cluster.
    std::map<const MCParticle *, const Cluster *> mcToLargestClusterMap;
    std::map<const MCParticle *, ClusterList> mcToAllClustersMap;
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
                mcToAllClustersMap[mc] = {cluster};
                continue;
            }

            const Cluster *currentLargestCluster = mcToLargestClusterMap[mc];

            if (cluster->GetNCaloHits() > currentLargestCluster->GetNCaloHits())
                mcToLargestClusterMap[mc] = cluster;

            mcToAllClustersMap[mc].push_back(cluster);
        }
        catch (const StatusCodeException &)
        {
            continue;
        }
    }

    int nFailed = 0;
    int nPassed = 0;

    // std::ofstream csvFile;
    // csvFile.open(fileName + ".csv");

    for (auto const &cluster : *clusters)
    {

        // ROOT TTree variable setup
        double failedHits = 0.0;
        double matchesMain = 0.0;
        double energyFromMain = 0.0;
        double clusterMainMCId = -999.0;
        double isLargestForMC = 0.0;
        double totalEnergyForCluster = 0.0;

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
        double energyOfMc = -999.0;

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
            energyOfMc = 0.0;

            for (auto mcHit : mcToCaloHit->second)
                // TODO: Lookup which stream calo hits are in
                energyOfMc += mcHit->GetElectromagneticEnergy();

            ++nPassed;
        }
        else
        {
            ++nFailed;
        }

        const int cId = cluster->GetParticleId();
        const int isShower = std::abs(cId) == MU_MINUS ? 0 : 1;
        const double isLargestShower = (isShower && cluster->GetNCaloHits() == largestShower) ? 1.0 : 0.0;

        // // Write out the CSV file whilst building up info for the ROOT TTree.
        // csvFile << "X, Z, Type, PID, IsAvailable, IsShower, MCId, IsIsolated, isVertex" << std::endl;

        // Get the vertex "list" which seems to only be used for the first element, if at all.
        // Write that out first.
        try
        {
            const VertexList *pVertexList(nullptr);
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));

            if (pVertexList == nullptr)
                throw STATUS_CODE_NOT_FOUND;

            // for (const auto vertex : *pVertexList)
            // {
            //     const CartesianVector pos = vertex->GetPosition();

            //     csvFile << pos.GetX() << ", " << pos.GetZ() << ", " << m_recoStatus << ", " << cluster->GetParticleId() << ", "
            //             << cluster->IsAvailable() << ", 0, -999, 0, 1" << std::endl;
            // }
        }
        catch (StatusCodeException)
        {
        }

        unsigned int index = 0;
        for (const auto caloHit : clusterCaloHits)
        {
            totalEnergyForCluster += caloHit->GetElectromagneticEnergy();

            // const CartesianVector pos = caloHit->GetPositionVector();
            const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);
            // const bool isIsolated = index >= (clusterCaloHits.size() - cluster->GetIsolatedCaloHitList().size());
            // int hitMCId = -999;

            if (it2 == eventLevelCaloHitToMCMap.end())
            {
                ++failedHits;
            }
            else
            {
                const auto mc = it2->second;
                // hitMCId = this->GetIdForMC(mc, mcIDMap);

                if (mc == pMCParticle)
                {
                    ++matchesMain;
                    energyFromMain += caloHit->GetElectromagneticEnergy();
                }
            }

            // csvFile << pos.GetX() << ", " << pos.GetZ() << ", " << m_recoStatus << ", " << cluster->GetParticleId() << ", "
                    // << cluster->IsAvailable() << ", " << isShower << ", " << hitMCId << ", " << isIsolated << ", "
                    // << "0" << std::endl;
            ++index;
        }

        // Finally, calculate the completeness and purity, and write out TTree.
        const double hitCompleteness = matchesMain / hitsInMC;
        const double hitPurity = matchesMain / clusterCaloHits.size();

        const double energyCompleteness = energyFromMain / energyOfMc;
        const double energyPurity = energyFromMain / totalEnergyForCluster;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "clusterNumber", (double)clusters->size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "mcID", clusterMainMCId));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "completeness", hitCompleteness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "purity", hitPurity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "energyCompleteness", energyCompleteness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "energyPurity", energyPurity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "numberOfHits", (double)clusterCaloHits.size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "mcEnergy", energyOfMc));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "failedHits", failedHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "isShower", (double)isShower));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "tsIDCorrect", this->IsTaggedCorrectly(cId, clusterMainMCId)));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "isLargestForMC", isLargestForMC));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), clusterTree, "isLargestShower", isLargestShower));
        PANDORA_MONITORING_API(FillTree(this->GetPandora(), clusterTree));
    }

    // Also have a higher level tree, that is flipped.
    // With the cluster level tree, we have "This cluster is X% complete and X% pure".
    // This is a higher level tree, so "This MC Particle is spread across X clusters, with X purity".
    for (auto mcToList : eventLevelMCToCaloHitMap)
    {
        auto mc = mcToList.first;
        auto mcCaloHits = mcToList.second;
        auto mcClusters = mcToAllClustersMap[mc];

        double hitsInMC = mcCaloHits.size();

        // TODO: Include hits split by view
        double numOfHitsInLargestCluster = 0;
        double matchesInLargest = 0;
        double totalHits = 0;
        std::vector<double> completeness;
        std::vector<double> purity;
        std::vector<double> numHitsInCluster;

        for (auto cluster : mcClusters)
        {
            double numOfHits = 0;
            double matchesMain = 0;

            // Get all calo hits for this cluster.
            CaloHitList clusterCaloHits;
            for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList())
            {
                CaloHitList hitsForCluster(*clusterHitPair.second);
                clusterCaloHits.merge(hitsForCluster);
            }
            CaloHitList isolatedHits = cluster->GetIsolatedCaloHitList();
            clusterCaloHits.merge(isolatedHits);

            for (auto hit : clusterCaloHits)
            {
                ++numOfHits;
                ++totalHits;

                if (eventLevelCaloHitToMCMap.count(hit) == 0)
                    continue;

                auto hitMc = eventLevelCaloHitToMCMap[hit];
                if (hitMc == mc)
                    ++matchesMain;
            }

            if (numOfHits > numOfHitsInLargestCluster)
            {
                numOfHitsInLargestCluster = numOfHits;
                matchesInLargest = matchesMain;
            }

            completeness.push_back(matchesMain / hitsInMC);
            purity.push_back(matchesMain / numOfHits);
            numHitsInCluster.push_back(numOfHits);
        }

        if (completeness.size() == 0)
        {
            completeness.push_back(0.0);
            purity.push_back(0.0);
        }

        const double completenessForLargestCluster = matchesInLargest > 0 ? matchesInLargest / hitsInMC : 0.0;
        const double purityForLargestCluster = matchesInLargest > 0 ? matchesInLargest / numOfHitsInLargestCluster : 0.0;

        double avgCompletness = 0.0;
        for (auto comp : completeness)
            avgCompletness += comp;
        avgCompletness = avgCompletness != 0 ? avgCompletness / completeness.size() : 0.0;

        double avgPurity = 0.0;
        for (auto pur : purity)
            avgPurity += pur;
        avgPurity = avgPurity != 0 ? avgPurity / purity.size() : 0.0;

        const int mcId = mc->GetParticleId();
        const int isShower = (std::abs(mcId) == E_MINUS) || (mcId == PHOTON) ? 0 : 1;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "mcID", (double)mc->GetParticleId()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "isShower", (double)isShower));

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "completeness", &completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "avgCompleteness", avgCompletness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "completenessOfLargest", completenessForLargestCluster));

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "purity", &purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "avgPurity", avgPurity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "purityOfLargest", purityForLargestCluster));

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "numberOfClusters", (double)mcClusters.size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "numberOfHitsInCluster", &numHitsInCluster));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "numberOfHits", numOfHitsInLargestCluster));

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "totalHitsOverAllClusters", totalHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "mcNumOfHits", hitsInMC));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "mcEnergy", (double)mc->GetEnergy()));

        PANDORA_MONITORING_API(FillTree(this->GetPandora(), mcTree));

        if (completenessForLargestCluster == 0 && hitsInMC > 10)
        {
            std::cout << mcId << " cluster was entirely missing!" << std::endl;
            std::cout << "It has " << hitsInMC << " hits, but 0 clusters" << std::endl;
        }
    }

    // Save the two trees.
    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), clusterTree, fileName + ".root", "UPDATE"));
    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), mcTree, fileName + ".root", "UPDATE"));

    PANDORA_MONITORING_API(Delete(this->GetPandora()));

    std::cout << " >> Failed to find MC in MC -> Calo map for " << nFailed << " / " << nPassed + nFailed << std::endl;
    // csvFile.close();

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::DumpRecoInfo(const ClusterList *clusters, const std::string &fileName) const
{
    const std::string recoTree = "recoInfoTree";
    PANDORA_MONITORING_API(Create(this->GetPandora()));

    const VertexList *pVertexList(nullptr);
    (void)PandoraContentApi::GetCurrentList(*this, pVertexList);

    if (!pVertexList || pVertexList->empty())
        return;

    const Vertex *pInteractionVetex = pVertexList->front();
    for (const Vertex *pVertex : *pVertexList)
        if ((pVertex->GetVertexLabel() == VERTEX_INTERACTION) && (pVertex->GetVertexType() == VERTEX_3D))
            pInteractionVetex = pVertex;

    const CartesianVector vertexPosition(pInteractionVetex->GetPosition());
    int nEntries = 0;

    // Populate PFO list + map to check particle tag for test beam only tag.
    const PfoList *pPfoList = NULL;
    std::map<const Cluster *, const ParticleFlowObject *> clusterToPfoMap;

    (void)PandoraContentApi::GetCurrentList(*this, pPfoList);

    if (pVertexList && ! pVertexList->empty())
    {
        for (auto pfo : *pPfoList)
        {
            for (auto cluster : pfo->GetClusterList())
                clusterToPfoMap.insert({cluster, pfo});
        }
    }

    double largestShower = 0;
    for (auto const &cluster : *clusters)
    {
        if (clusterToPfoMap.count(cluster) == 0 || ! LArPfoHelper::IsTestBeam(clusterToPfoMap[cluster]))
            continue;

        if (std::abs(cluster->GetParticleId()) != MU_MINUS && cluster->GetNCaloHits() > largestShower)
            largestShower = cluster->GetNCaloHits();
    }

    for (auto const &cluster : *clusters)
    {
        if (std::abs(cluster->GetParticleId()) == MU_MINUS)
           continue;

        if (cluster->GetNCaloHits() < 3)
            continue;

        // For the shower-like clusters, build up some information about them.
        std::vector<pandora::CartesianVector> cartesianPointVector;

        for (auto const &hitList : cluster->GetOrderedCaloHitList())
            for (auto const &hit : *(hitList.second))
                cartesianPointVector.push_back(hit->GetPositionVector());

        const LArShowerPCA initialLArShowerPCA(lar_content::LArPfoHelper::GetPrincipalComponents(cartesianPointVector, vertexPosition));

        const pandora::CartesianVector& centroid(initialLArShowerPCA.GetCentroid());
        const pandora::CartesianVector& primaryAxis(initialLArShowerPCA.GetPrimaryAxis());
        const pandora::CartesianVector& secondaryAxis(initialLArShowerPCA.GetSecondaryAxis());
        const pandora::CartesianVector& tertiaryAxis(initialLArShowerPCA.GetTertiaryAxis());
        const pandora::CartesianVector& eigenValues(initialLArShowerPCA.GetEigenValues());

        const pandora::CartesianVector projectedVertexPosition(centroid - (primaryAxis.GetUnitVector() * (centroid - vertexPosition).GetDotProduct(primaryAxis)));
        const float testProjection(primaryAxis.GetDotProduct(projectedVertexPosition - centroid));
        const float directionScaleFactor((testProjection > std::numeric_limits<float>::epsilon()) ? -1.f : 1.f);

        const lar_content::LArShowerPCA larShowerPCA(centroid,
                                                     primaryAxis * directionScaleFactor,
                                                     secondaryAxis * directionScaleFactor,
                                                     tertiaryAxis * directionScaleFactor,
                                                     eigenValues);

        const pandora::CartesianVector& showerLength(larShowerPCA.GetAxisLengths());
        const pandora::CartesianVector& showerDirection(larShowerPCA.GetPrimaryAxis());

        double length(showerLength.GetX());
        double openingAngle(larShowerPCA.GetPrimaryLength() > 0.f ?
                                 std::atan(larShowerPCA.GetSecondaryLength() / larShowerPCA.GetPrimaryLength()):
                                 0.f);

        const HitType hitType(LArClusterHelper::GetClusterHitType(cluster));
        double viewTypeDouble = -1.0;

        switch (hitType) {
            case pandora::TPC_VIEW_U: viewTypeDouble = 0.0; break;
            case pandora::TPC_VIEW_V: viewTypeDouble = 1.0; break;
            case pandora::TPC_VIEW_W: viewTypeDouble = 2.0; break;
            case pandora::TPC_3D: viewTypeDouble = 3.0; break;
            default: viewTypeDouble = -1.0;
        }

        auto pfo = clusterToPfoMap.count(cluster) > 0 ? clusterToPfoMap[cluster] : NULL;
        double isTestBeam = pfo && LArPfoHelper::IsTestBeam(pfo) ? 1.0 : 0.0;
        double isTestBeamFinal = pfo && LArPfoHelper::IsTestBeamFinalState(pfo) ? 1.0 : 0.0;

        const double isLargestShower = cluster->GetNCaloHits() == largestShower ? 1.0 : 0.0;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoNumberOfHits", (double)cluster->GetNCaloHits()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoView", viewTypeDouble));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoIsLargestShower", isLargestShower));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoIsTestBeam", isTestBeam));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoIsTestBeamFinal", isTestBeamFinal));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoShowerLength", length));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoShowerOpeningAngle", openingAngle));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoShowerDirectionX", (double)showerDirection.GetX()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoShowerDirectionY", (double)showerDirection.GetY()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), recoTree, "recoShowerDirectionZ", (double)showerDirection.GetZ()));
        PANDORA_MONITORING_API(FillTree(this->GetPandora(), recoTree));

        ++nEntries;
    }

    if (nEntries > 0)
        PANDORA_MONITORING_API(SaveTree(this->GetPandora(), recoTree, fileName + ".root", "UPDATE"));

    PANDORA_MONITORING_API(Delete(this->GetPandora()));

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::GetMCMaps(const ClusterList * /*clusterList*/, const std::string &clusterListName,
    LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, LArMCParticleHelper::MCContributionMap &MCtoCaloMap,
    const MCParticleList *mcParticleList) const
{
    try
    {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "Input", mcParticleList));
    }
    catch (StatusCodeException e)
    {
        std::cout << "Failed to get MCParticleList: " << e.ToString() << std::endl;
        return;
    }

    LArMCParticleHelper::MCRelationMap mcToTargetMCMap;
    LArMCParticleHelper::GetMCToSelfMap(mcParticleList, mcToTargetMCMap);

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

double ClusterDumpingAlgorithm::GetIdForMC(const MCParticle *mc, std::map<const MCParticle *, int> &idMap) const
{
    if (idMap.count(mc) == 0)
        idMap[mc] = idMap.size();

    return idMap[mc];
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ClusterDumpingAlgorithm::IsTaggedCorrectly(const int cId, const int mcId) const
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

StatusCode ClusterDumpingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "DumpClusters", m_dumpClusterList));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "RecoStatus", m_recoStatus));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "NonSplitMode", m_nonSplit));
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "InputViewNames", m_viewNames));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_content
