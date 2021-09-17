/**
 *  @file   larpandoracontent/LArUtility/ClusterDumpingAlgorithm.cc
 *
 *  @brief  Implementation for the debug cluster dumping algorithm class.
 *
 *  $Log: $
 */

#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"
#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArUtility/ClusterDumpingAlgorithm.h"

#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArMvaHelper.h"
#include "larpandoracontent/LArHelpers/LArVertexHelper.h"

#include "larpandoracontent/LArObjects/LArTwoDSlidingFitResult.h"

#include <chrono>
#include <fstream>

#include <Eigen/Dense>

using namespace pandora;

namespace lar_content
{

StatusCode ClusterDumpingAlgorithm::Run()
{
    for (const std::string &listName : m_clusterListNames)
    {

        const ClusterList *pClusterList = nullptr;

        try
        {
            PANDORA_THROW_RESULT_IF_AND_IF(
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
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::DumpClusterList(const ClusterList *clusters, const std::string &clusterListName) const
{
    // Pick folder.
    const std::string data_folder = "/home/ryan/git/data/showers/pandoraOutputs";
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

            const Cluster *currentCluster = mcToLargestClusterMap[mc];

            if (cluster->GetNCaloHits() > currentCluster->GetNCaloHits())
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
            totalEnergyForCluster += caloHit->GetElectromagneticEnergy();

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
                    energyFromMain += caloHit->GetElectromagneticEnergy();
                }
            }

            csvFile << pos.GetX() << ", " << pos.GetZ() << ", " << m_recoStatus << ", " << cluster->GetParticleId() << ", "
                    << cluster->IsAvailable() << ", " << isShower << ", " << hitMCId << ", " << isIsolated << ", "
                    << "0" << std::endl;
            ++index;
        }

        // Finally, calculate the completeness and purity, and write out TTree.
        const double hitCompleteness = matchesMain / hitsInMC;
        const double hitPurity = matchesMain / clusterCaloHits.size();

        std::cout << "Total Energy for Cluster: " << totalEnergyForCluster << std::endl;
        std::cout << "Total Matched Energy for Cluster: " << energyFromMain << std::endl;
        std::cout << "Energy of MC: " << energyOfMc << std::endl;
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

        double numOfHitsInLargestCluster = 0;
        double matchesInLargest = 0;
        double totalHits = 0;
        std::vector<double> completeness;
        std::vector<double> purity;

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
        }

        if (completeness.size() == 0)
        {
            completeness.push_back(0.0);
            purity.push_back(0.0);
        }

        const double completenessForLargestCluster = matchesInLargest > 0 ? matchesInLargest / hitsInMC : 0.0;
        const double purityForLargestCluster = matchesInLargest > 0 ? matchesInLargest / numOfHitsInLargestCluster : 0.0;
        const int mcId = mc->GetParticleId();
        const int isShower = std::abs(mcId) == MU_MINUS ? 0 : 1;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "mcID", (double)mc->GetParticleId()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "isShower", (double)isShower));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "completeness", &completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "purity", &purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "completenessOfLargest", completenessForLargestCluster));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "purityOfLargest", purityForLargestCluster));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "numberOfClusters", (double)mcClusters.size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "numberOfHits", numOfHitsInLargestCluster));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "totalHitsOverAllClusters", totalHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), mcTree, "mcNumOfHits", hitsInMC));
        PANDORA_MONITORING_API(FillTree(this->GetPandora(), mcTree));
    }

    // Save the two trees.
    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), clusterTree, fileName + ".root", "RECREATE"));
    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), mcTree, fileName + ".root", "UPDATE"));

    PANDORA_MONITORING_API(Delete(this->GetPandora()));

    std::cout << " >> Failed to find MC in MC -> Calo map for " << nFailed << " / " << nPassed + nFailed << std::endl;
    csvFile.close();

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::GetMCMaps(const ClusterList * /*clusterList*/, const std::string &clusterListName,
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
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadVectorOfValues(xmlHandle, "InputClusterListNames", m_clusterListNames));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_content
