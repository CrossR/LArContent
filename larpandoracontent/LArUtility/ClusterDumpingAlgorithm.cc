/**
 *  @file   larpandoracontent/LArUtility/ClusterDumpingAlgorithm.cc
 *
 *  @brief  Implementation for the debug cluster dumping algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"
#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"

#include "larpandoracontent/LArUtility/ClusterDumpingAlgorithm.h"

#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include <fstream>

using namespace pandora;

namespace lar_content
{

StatusCode ClusterDumpingAlgorithm::Run()
{
    for (const std::string &listName : m_clusterListNames)
        this->DumpClusterList(listName);

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::DumpClusterList(const std::string &clusterListName) const
{

    // Pick folder.
    const std::string data_folder = "/home/scratch/showerClusters";
    system(("mkdir -p " + data_folder).c_str());

    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    while (true)
    {
        fileName = data_folder + "/clusters_" +
            clusterListName + "_" + m_recoStatus + "_" +
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

    if (pClusterList == nullptr || pClusterList->size() == 0) {
        std::cout << "Cluster list was empty." << std::endl;
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
    //
    // Get every single CaloHit, then use the helper to build
    // the two maps.
    LArMCParticleHelper::MCRelationMap mcToTargetMCMap;
    LArMCParticleHelper::GetMCToSelfMap(pMCParticleList, mcToTargetMCMap);

    LArMCParticleHelper::CaloHitToMCMap eventLevelCaloHitToMCMap;
    LArMCParticleHelper::MCContributionMap eventLevelMCToCaloHitMap;

    CaloHitList caloHits;
    for (auto const &cluster : *pClusterList) {
        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            CaloHitList hitsForCluster(*clusterHitPair.second);
            caloHits.merge(hitsForCluster);
        }

        CaloHitList isolatedHits = cluster->GetIsolatedCaloHitList();
        caloHits.merge(isolatedHits);
    }

    try {
        LArMCParticleHelper::GetMCParticleToCaloHitMatches(
            &(caloHits), mcToTargetMCMap, eventLevelCaloHitToMCMap, eventLevelMCToCaloHitMap
        );
    } catch (StatusCodeException e) {
        std::cout << "Failed to get matches: " << e.ToString() << std::endl;
    }
    caloHits.clear();

    const std::string treeName = "showerClustersTree";
    PANDORA_MONITORING_API(Create(this->GetPandora()));

    for (auto const &cluster : *pClusterList) {

        // ROOT TTree variable setup
        float failedHits = 0.0;
        float matchesMain = 0.0;
        int clusterMainMCId = -999;

        // Get all calo hits for this cluster.
        CaloHitList clusterCaloHits;
        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            CaloHitList hitsForCluster(*clusterHitPair.second);
            clusterCaloHits.merge(hitsForCluster);
        }
        CaloHitList isolatedHits = cluster->GetIsolatedCaloHitList();
        clusterCaloHits.merge(isolatedHits);

        const MCParticle *pMCParticle = nullptr;
        float hitsInMC = -999;

        try {
            pMCParticle = MCParticleHelper::GetMainMCParticle(cluster);
            clusterMainMCId = pMCParticle->GetParticleId();
        } catch (const StatusCodeException &) {
            // TODO: Attach debugger and check why!
            int c_size = cluster->GetOrderedCaloHitList().size() + cluster->GetIsolatedCaloHitList().size();
            std::cout << "  ## No MC. Size " << c_size << std::endl;
        }

        auto mcToCaloHit = eventLevelMCToCaloHitMap.find(pMCParticle);
        if (mcToCaloHit != eventLevelMCToCaloHitMap.end()) {
            hitsInMC = mcToCaloHit->second.size();
        } else {
            std::cout << " >> Failed to find MC in MC -> Calo map!" << std::endl;
        }

        const int cId = cluster->GetParticleId();
        const int isShower = std::abs(cId) == MU_MINUS ? 0 : 1;

        // Write out the CSV file whilst building up info for the ROOT TTree.
        // TODO: Get Vertex List and dump that out to CSV too.
        csvFile << "X, Z, Type, PID, IsAvailable, IsShower, MCId, IsIsolated" << std::endl;

        int index = 0;
        for (const auto caloHit : clusterCaloHits) {

            const CartesianVector pos = caloHit->GetPositionVector();
            const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);
            const bool isIsolated = index >= (clusterCaloHits.size() - cluster->GetIsolatedCaloHitList().size());
            int hitMCId = -999;

            if (it2 == eventLevelCaloHitToMCMap.end()) {
               ++failedHits;
            } else {
                const auto mc = it2->second;
                hitMCId = *(int *) &mc;

                if (mc == pMCParticle) {
                    ++matchesMain;
                }
            }

            csvFile << pos.GetX() << ", "
                    << pos.GetZ() << ", "
                    << m_recoStatus << ", "
                    << cluster->GetParticleId() << ", "
                    << cluster->IsAvailable() << ", "
                    << isShower << ", "
                    << hitMCId << ", "
                    << isIsolated
                    << std::endl;
            ++index;
        }

        // Finally, calculaate the completeness and purity, and write out TTree.
        const float completeness = matchesMain / hitsInMC;
        const float purity = matchesMain / clusterCaloHits.size();

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "clusterNumber", (int) pClusterList->size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "completeness", completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "purity", purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "numberOfHits", (int) clusterCaloHits.size()));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "failedHits", failedHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "mcID", clusterMainMCId));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "isShower", isShower));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "tsIDCorrect", this->IsTaggedCorrectly(cId, clusterMainMCId)));
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

int ClusterDumpingAlgorithm::IsTaggedCorrectly(const int cId, const int mcId) const {

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

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "RecoStatus", m_recoStatus));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "InputClusterListNames", m_clusterListNames));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_content
