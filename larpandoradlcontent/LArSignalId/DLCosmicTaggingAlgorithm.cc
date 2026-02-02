/**
 *  @file   larpandoradlcontent/LArSignalId/DLCosmicTaggingAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning cosmic hit tagging.
 *
 *  $Log: $
 */

#include <Objects/MCParticle.h>
#include <chrono>
#include <cmath>

#include <torch/script.h>
#include <torch/torch.h>

#include "larpandoracontent/LArObjects/LArCaloHit.h"

#include "larpandoracontent/LArObjects/LArCaloHit.h"
#include "larpandoracontent/LArHelpers/LArMvaHelper.h"

#include "larpandoradlcontent/LArSignalId/DLCosmicTaggingAlgorithm.h"
#include "larpandoradlcontent/LArVertex/DlVertexingAlgorithm.h"

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DLCosmicTaggingAlgorithm::DLCosmicTaggingAlgorithm(){}

DLCosmicTaggingAlgorithm::~DLCosmicTaggingAlgorithm()
{
    if (m_writeTree)
    {
        try
        {
            PANDORA_MONITORING_API(SaveTree(this->GetPandora(), m_rootTreeName, m_rootFileName, "RECREATE"));
        }
        catch (StatusCodeException e)
        {
            std::cout << "DLCosmicTaggingAlgorithm: Unable to write to ROOT tree" << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DLCosmicTaggingAlgorithm::Run()
{
    // 3 Output classes, NULL, Nu, Cosmic.
    m_nClasses = 3;

    if (m_trainingMode)
        return this->PrepareTrainingSample();
    else
        return this->Infer();

    return STATUS_CODE_SUCCESS;
}

StatusCode DLCosmicTaggingAlgorithm::PrepareTrainingSample()
{
    const CaloHitList *pCaloHitList2D(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "CaloHitList2D", pCaloHitList2D));
    const MCParticleList *pMCParticleList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pMCParticleList));
    LArMCParticleHelper::MCContributionMap mcToHitsMap;
    LArMCParticleHelper::GetMCToHitsMap(pCaloHitList2D, pMCParticleList, mcToHitsMap);
    MCParticleList hierarchy;
    LArMCParticleHelper::CompleteMCHierarchy(mcToHitsMap, hierarchy);

    if (m_visualise)
    {
        PANDORA_MONITORING_API(SetEveDisplayParameters(this->GetPandora(), true, DETECTOR_VIEW_XZ, -1.f, 1.f, 1.f));

        const CaloHitList *caloU{nullptr};
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "CaloHitListU", caloU));
        const CaloHitList *caloV{nullptr};
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "CaloHitListV", caloV));
        const CaloHitList *caloW{nullptr};
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "CaloHitListW", caloW));

        PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), caloU, "Calo U", GRAY));
        PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), caloV, "Calo V", GRAY));
        PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), caloW, "Calo W", GRAY));
        PANDORA_MONITORING_API(ViewEvent(this->GetPandora()));
    }

    // Get boundaries for hits and make x dimension common
    std::map<HitType, float> wireMin, wireMax;
    float driftMin{std::numeric_limits<float>::max()}, driftMax{-std::numeric_limits<float>::max()};
    for (const std::string &listname : m_caloHitListNames)
    {
        const CaloHitList *pCaloHitList{nullptr};
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, listname, pCaloHitList));
        if (pCaloHitList->empty())
            continue;

        HitType view{pCaloHitList->front()->GetHitType()};
        float viewDriftMin{driftMin}, viewDriftMax{driftMax};
        this->GetHitRegion(*pCaloHitList, viewDriftMin, viewDriftMax, wireMin[view], wireMax[view]);
        driftMin = std::min(viewDriftMin, driftMin);
        driftMax = std::max(viewDriftMax, driftMax);
    }

    for (const std::string &listname : m_caloHitListNames)
    {
        const CaloHitList *pCaloHitList(nullptr);
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, listname, pCaloHitList));
        if (pCaloHitList->empty())
            continue;

        HitType view{pCaloHitList->front()->GetHitType()};
        const bool isU{view == TPC_VIEW_U}, isV{view == TPC_VIEW_V}, isW{view == TPC_VIEW_W};
        if (!(isU || isV || isW))
            return STATUS_CODE_NOT_ALLOWED;

        std::map<const CaloHit*, const MCParticle*> caloHitToMCMap;

        for (const MCParticle *mc : hierarchy)
            for (const CaloHit* hit : mcToHitsMap[mc])
                caloHitToMCMap.insert({hit, mc});

        if (caloHitToMCMap.empty())
            continue;

        const std::string trainingFilename{m_trainingOutputFile + "_" + listname + ".csv"};
        unsigned long nHits{0};
        const unsigned int nuance{LArMCParticleHelper::GetNuanceCode(hierarchy.front())};

        // Calo hits
        double xMin{driftMin}, xMax{driftMax}, zMin{wireMin[view]}, zMax{wireMax[view]};

        LArMvaHelper::MvaFeatureVector featureVector;
        featureVector.emplace_back(static_cast<double>(nuance));
        // Retain the hit region
        featureVector.emplace_back(xMin);
        featureVector.emplace_back(xMax);
        featureVector.emplace_back(zMin);
        featureVector.emplace_back(zMax);

        for (const CaloHit *pCaloHit : *pCaloHitList)
        {
            const float x{pCaloHit->GetPositionVector().GetX()}, z{pCaloHit->GetPositionVector().GetZ()}, adc{pCaloHit->GetMipEquivalentEnergy()};
            const auto mc(caloHitToMCMap.find(pCaloHit));
            const float pdg(mc != caloHitToMCMap.end() ? static_cast<float>(mc->second->GetParticleId()) : 0.f);
            featureVector.emplace_back(static_cast<double>(x));
            featureVector.emplace_back(static_cast<double>(z));
            featureVector.emplace_back(static_cast<double>(adc));
            featureVector.emplace_back(static_cast<double>(pdg));
            ++nHits;
        }

        featureVector.insert(featureVector.begin() + 5, static_cast<double>(nHits));

        // Only write out the feature vector if there were enough hits in the region of interest
        if (nHits > 10)
            LArMvaHelper::ProduceTrainingExample(trainingFilename, true, featureVector);
    }

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DLCosmicTaggingAlgorithm::Infer()
{
    std::map<HitType, float> wireMin, wireMax;
    float driftMin{std::numeric_limits<float>::max()}, driftMax{-std::numeric_limits<float>::max()};

    for (const std::string &listname : m_caloHitListNames)
    {
        const CaloHitList *pCaloHitList{nullptr};
        PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, listname, pCaloHitList));

        if (pCaloHitList == nullptr || pCaloHitList->empty())
            continue;

        HitType view{pCaloHitList->front()->GetHitType()};
        float viewDriftMin{driftMin}, viewDriftMax{driftMax};
        this->GetHitRegion(*pCaloHitList, viewDriftMin, viewDriftMax, wireMin[view], wireMax[view]);
        driftMin = std::min(viewDriftMin, driftMin);
        driftMax = std::max(viewDriftMax, driftMax);
    }

    for (const std::string &listName : m_caloHitListNames)
    {
        const CaloHitList *pCaloHitList{nullptr};
        PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, listName, pCaloHitList));

        if (pCaloHitList == nullptr || pCaloHitList->empty())
            continue;

        HitType view{pCaloHitList->front()->GetHitType()};
        const bool isU{view == TPC_VIEW_U}, isV{view == TPC_VIEW_V}, isW{view == TPC_VIEW_W};
        if (!isU && !isV && !isW)
            return STATUS_CODE_NOT_ALLOWED;

        LArDLHelper::TorchInput input;
        PixelVector pixelVector;
        CaloHitToPixelMap caloHitToPixelMap;
        this->MakeNetworkInputFromHits(*pCaloHitList, view, driftMin, driftMax, wireMin[view], wireMax[view], input, pixelVector, caloHitToPixelMap);

        // Run the input through the trained model
        LArDLHelper::TorchInputVector inputs;
        inputs.push_back(input);
        LArDLHelper::TorchOutput output;

        if (isU)
            LArDLHelper::Forward(m_modelU, inputs, output);
        else if (isV)
            LArDLHelper::Forward(m_modelV, inputs, output);
        else
            LArDLHelper::Forward(m_modelW, inputs, output);

        // the argmax result is a 1 x height x width tensor where each element is a class id
        auto classesAccessor{output.accessor<float, 4>()};

        CaloHitList backgroundHits, neutrinoHits, cosmicRayHits;
        for (const CaloHit *pCaloHit : *pCaloHitList)
        {
            auto found{caloHitToPixelMap.find(pCaloHit)};
            if (found == caloHitToPixelMap.end())
                continue;
            auto pixelMap = found->second;

            const int pixelZ(std::get<0>(pixelMap));
            const int pixelX(std::get<1>(pixelMap));

            // Apply softmax to loss to get actual probability
            float probNull = classesAccessor[0][0][pixelZ][pixelX];
            float probNeutrino = classesAccessor[0][1][pixelZ][pixelX];
            float probCosmic = classesAccessor[0][2][pixelZ][pixelX];

            if (probNeutrino > probCosmic && probNeutrino > probNull)
                neutrinoHits.push_back(pCaloHit);
            else if (probCosmic > probNeutrino && probCosmic > probNull)
                cosmicRayHits.push_back(pCaloHit);
            else
                backgroundHits.push_back(pCaloHit);

            float recipSum = 1.f / (probNeutrino + probCosmic);
            // Adjust probabilities to ignore null hits and update LArCaloHit
            probNeutrino *= recipSum;
            probCosmic *= recipSum;

            // TODO: This std::map<std::string, float> interface isn't ideal...
            //       Would it be better to have it typed in some capacity to remove (enum or similar?), to remove typo issues?
            // TODO: Generic enough names here? BeamProbabilty for PD instead?
            LArCaloHit *pLArCaloHit{const_cast<LArCaloHit *>(dynamic_cast<const LArCaloHit *>(pCaloHit))};
            pLArCaloHit->SetProperty("NeutrinoProbability", probNeutrino);
            pLArCaloHit->SetProperty("CosmicProbability", probCosmic);
        }

        if (m_visualise)
        {
            const std::string neutrinoListName("NeutrinoHits_" + listName);
            const std::string cosmicListName("CosmicHits_" + listName);
            const std::string backgroundListName("BackgroundHits_" + listName);

            PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), &neutrinoHits, neutrinoListName, BLUE));
            PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), &cosmicRayHits, cosmicListName, RED));
            PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), &backgroundHits, backgroundListName, BLACK));
        }
    }

    if (m_visualise)
    {
        PANDORA_MONITORING_API(ViewEvent(this->GetPandora()));
    }

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DLCosmicTaggingAlgorithm::MakeNetworkInputFromHits(const CaloHitList &caloHits, const HitType view, const float xMin,
    const float xMax, const float zMin, const float zMax, LArDLHelper::TorchInput &networkInput, PixelVector &pixelVector,
    CaloHitToPixelMap &caloHitToPixelMap) const
{
    // ATTN If wire w pitches vary between TPCs, exception will be raised in initialisation of lar pseudolayer plugin
    const LArTPC *const pTPC(this->GetPandora().GetGeometry()->GetLArTPCMap().begin()->second);
    const float pitch(view == TPC_VIEW_U ? pTPC->GetWirePitchU() : view == TPC_VIEW_V ? pTPC->GetWirePitchV() : pTPC->GetWirePitchW());

    // Determine the bin edges
    std::vector<double> xBinEdges(m_width + 1);
    std::vector<double> zBinEdges(m_height + 1);
    xBinEdges[0] = xMin - 0.5f * m_driftStep;
    const double dx = ((xMax + 0.5f * m_driftStep) - xBinEdges[0]) / m_width;
    for (int i = 1; i < m_width + 1; ++i)
        xBinEdges[i] = xBinEdges[i - 1] + dx;
    zBinEdges[0] = zMin - 0.5f * pitch;
    const double dz = ((zMax + 0.5f * pitch) - zBinEdges[0]) / m_height;
    for (int i = 1; i < m_height + 1; ++i)
        zBinEdges[i] = zBinEdges[i - 1] + dz;

    LArDLHelper::InitialiseInput({1, 1, m_height, m_width}, networkInput);
    auto accessor = networkInput.accessor<float, 4>();

    for (const CaloHit *pCaloHit : caloHits)
    {
        const float x{pCaloHit->GetPositionVector().GetX()};
        const float z{pCaloHit->GetPositionVector().GetZ()};
        if (m_pass > 1)
        {
            if (x < xMin || x > xMax || z < zMin || z > zMax)
                continue;
        }
        const float adc{pCaloHit->GetMipEquivalentEnergy()};
        const int pixelX{static_cast<int>(std::floor((x - xBinEdges[0]) / dx))};
        const int pixelZ{static_cast<int>(std::floor((z - zBinEdges[0]) / dz))};
        accessor[0][0][pixelZ][pixelX] += adc;

        caloHitToPixelMap.insert({pCaloHit, std::make_tuple(pixelZ, pixelX)});
    }
    for (int row = 0; row < m_height; ++row)
    {
        for (int col = 0; col < m_width; ++col)
        {
            const float value{accessor[0][0][row][col]};
            if (value > 0)
                pixelVector.emplace_back(std::make_pair(row, col));
        }
    }

    return STATUS_CODE_SUCCESS;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

StatusCode DLCosmicTaggingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, DlVertexingBaseAlgorithm::ReadSettings(xmlHandle));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "Visualise", m_visualise));

    if (!m_trainingMode)
    {
        PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "WriteTree", m_writeTree));
        if (m_writeTree)
        {
            PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "RootTreeName", m_rootTreeName));
            PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "RootFileName", m_rootFileName));
        }
    }

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_dl_content
