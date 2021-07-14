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

#include <chrono>

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

        this->ProduceTrainingFile(pClusterList, listName);
    }

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

StatusCode DlShowerGrowingAlgorithm::Infer()
{
    return STATUS_CODE_SUCCESS;
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
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileNameU", m_modelFileNameU));
        m_modelFileNameU = LArFileHelper::FindFileInPath(m_modelFileNameU, "FW_SEARCH_PATH");
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, LArDLHelper::LoadModel(m_modelFileNameU, m_modelU));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileNameV", m_modelFileNameV));
        m_modelFileNameV = LArFileHelper::FindFileInPath(m_modelFileNameV, "FW_SEARCH_PATH");
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, LArDLHelper::LoadModel(m_modelFileNameV, m_modelV));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ModelFileNameW", m_modelFileNameW));
        m_modelFileNameW = LArFileHelper::FindFileInPath(m_modelFileNameW, "FW_SEARCH_PATH");
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, LArDLHelper::LoadModel(m_modelFileNameW, m_modelW));
    }

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadVectorOfValues(xmlHandle, "InputClusterListNames", m_clusterListNames));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "Visualize", m_visualize));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_dl_content
