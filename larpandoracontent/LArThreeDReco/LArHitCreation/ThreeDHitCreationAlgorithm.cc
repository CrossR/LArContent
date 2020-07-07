/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.cc
 *
 *  @brief  Implementation of the three dimensional hit creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

// TODO: Check over includes once metric stuff is deleted.
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArFileHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArObjectHelper.h"
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include "larpandoracontent/LArObjects/LArMCParticle.h"
#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"

#include "larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h"

#include <algorithm>
#include <fstream>
#include <sys/stat.h>

#ifdef MONITORING
#include "PandoraMonitoringApi.h"
#endif

using namespace pandora;

namespace lar_content
{

ThreeDHitCreationAlgorithm::ThreeDHitCreationAlgorithm() :
    m_metricFileName(""), // TODO: Remove
    m_metricTreeName("threeDTrackTree"), // TODO: Remove
    m_iterateTrackHits(true),
    m_iterateShowerHits(false),
    m_useRANSACMethod(false),
    m_slidingFitHalfWindow(10),
    m_nHitRefinementIterations(10),
    m_sigma3DFitMultiplier(0.2),
    m_iterationMaxChi2Ratio(1.),
    m_interpolationCutOff(10.)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::FilterCaloHitsByType(const CaloHitVector &inputCaloHitVector, const HitType hitType, CaloHitVector &outputCaloHitVector) const
{
    for (const CaloHit *const pCaloHit : inputCaloHitVector)
    {
        if (hitType == pCaloHit->GetHitType())
            outputCaloHitVector.push_back(pCaloHit);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeDHitCreationAlgorithm::Run()
{
    const PfoList *pPfoList(nullptr);
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, m_inputPfoListName, pPfoList));

    if (!pPfoList || pPfoList->empty())
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "ThreeDHitCreationAlgorithm: unable to find pfo list " << m_inputPfoListName << std::endl;

        return STATUS_CODE_SUCCESS;
    }

    CaloHitList allNewThreeDHits;
    ProtoHitVectorMap allProtoHitVectors;

    PfoVector pfoVector(pPfoList->begin(), pPfoList->end());
    std::sort(pfoVector.begin(), pfoVector.end(), LArPfoHelper::SortByNHits);

    for (const ParticleFlowObject *const pPfo : pfoVector)
    {
        ProtoHitVector protoHitVector;
        int numberOfFailedAlgorithms = 0;

        for (HitCreationBaseTool *const pHitCreationTool : m_algorithmToolVector)
        {
            CaloHitVector remainingTwoDHits;
            this->SeparateTwoDHits(pPfo, protoHitVector, remainingTwoDHits);

            if (remainingTwoDHits.empty())
                break;

            if (!m_useRANSACMethod) {
                // TODO: Drop try-catch, only needed to ensure metric generation.
                try {
                    pHitCreationTool->Run(this, pPfo, remainingTwoDHits, protoHitVector);
                } catch (StatusCodeException &statusCodeException) {
                    // std::vector<std::pair<std::string, ProtoHitVector>> allProtoHitsToPlot;
                    // this->OutputDebugMetrics(pPfo, protoHitVector, allProtoHitVectors, allProtoHitsToPlot);

                    throw statusCodeException;
                }
            } else {
                // TODO: Drop try-catch, only needed to ensure metric generation.
                try
                {
                    pHitCreationTool->Run(this, pPfo, remainingTwoDHits, protoHitVector);

                    if (LArPfoHelper::IsTrack(pPfo))
                    {
                        // TODO: Replace 10 with a configuration controlled number.
                        for (unsigned int i = 0; i < 10; ++i)
                        {
                            int sizeBefore = protoHitVector.size();
                            this->InterpolationMethod(pPfo, protoHitVector);
                            int sizeAfter = protoHitVector.size();

                            if (sizeBefore == sizeAfter)
                                break;
                        }

                        this->IterativeTreatment(protoHitVector);
                        allProtoHitVectors.insert(ProtoHitVectorMap::value_type(pHitCreationTool->GetInstanceName(), protoHitVector));
                        protoHitVector.clear();
                    }
                }
                catch (StatusCodeException &statusCodeException)
                {
                    std::cout << "Running tool " << pHitCreationTool->GetInstanceName()
                        << " failed with status code " << statusCodeException.ToString()
                        << std::endl;
                    ++numberOfFailedAlgorithms;

                    if (LArPfoHelper::IsTrack(pPfo))
                    {
                        allProtoHitVectors.insert(ProtoHitVectorMap::value_type(pHitCreationTool->GetInstanceName(), protoHitVector));
                        protoHitVector.clear();
                    }

                    continue;
                }
            }
        }

        if (numberOfFailedAlgorithms == m_algorithmToolVector.size())
        {
            // TODO: Remove metric code.
            // std::vector<std::pair<std::string, ProtoHitVector>> allProtoHitsToPlot;
            // this->OutputDebugMetrics(pPfo, protoHitVector, allProtoHitVectors, allProtoHitsToPlot);
            continue;
        }

        bool shouldUseIterativeTreatment = (
                (m_iterateTrackHits && LArPfoHelper::IsTrack(pPfo)) ||
                (m_iterateShowerHits && LArPfoHelper::IsShower(pPfo))
        );

        // ATTN: Skip for RANSAC, since it will be done later.
        if (shouldUseIterativeTreatment && !m_useRANSACMethod)
            this->IterativeTreatment(protoHitVector);

        if (m_useRANSACMethod && LArPfoHelper::IsTrack(pPfo))
        {
            this->ConsolidatedMethod(pPfo, allProtoHitVectors, protoHitVector);
            allProtoHitVectors.clear();
        }

        // TODO: Remove metric code.
        // std::vector<std::pair<std::string, ProtoHitVector>> allProtoHitsToPlot;
        // this->OutputDebugMetrics(pPfo, protoHitVector, allProtoHitVectors, allProtoHitsToPlot);

        if (protoHitVector.empty())
            continue;

        CaloHitList newThreeDHits;
        this->CreateThreeDHits(protoHitVector, newThreeDHits);
        this->AddThreeDHitsToPfo(pPfo, newThreeDHits);

        allNewThreeDHits.insert(allNewThreeDHits.end(), newThreeDHits.begin(), newThreeDHits.end());
    }

    if (!allNewThreeDHits.empty())
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList(*this, allNewThreeDHits, m_outputCaloHitListName));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::SeparateTwoDHits(const ParticleFlowObject *const pPfo, const ProtoHitVector &protoHitVector, CaloHitVector &remainingHitVector) const
{
    ClusterList twoDClusterList;
    LArPfoHelper::GetTwoDClusterList(pPfo, twoDClusterList);
    CaloHitList remainingHitList;

    for (const Cluster *const pCluster : twoDClusterList)
    {
        if (TPC_3D == LArClusterHelper::GetClusterHitType(pCluster))
            throw StatusCodeException(STATUS_CODE_FAILURE);

        pCluster->GetOrderedCaloHitList().FillCaloHitList(remainingHitList);
    }

    CaloHitSet remainingHitSet(remainingHitList.begin(), remainingHitList.end());

    for (const ProtoHit &protoHit : protoHitVector)
    {
        CaloHitSet::iterator eraseIter = remainingHitSet.find(protoHit.GetParentCaloHit2D());

        if (remainingHitSet.end() == eraseIter)
            throw StatusCodeException(STATUS_CODE_FAILURE);

        remainingHitSet.erase(eraseIter);
    }

    remainingHitVector.insert(remainingHitVector.end(), remainingHitSet.begin(), remainingHitSet.end());
    std::sort(remainingHitVector.begin(), remainingHitVector.end(), LArClusterHelper::SortHitsByPosition);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::IterativeTreatment(ProtoHitVector &protoHitVector) const
{
    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(m_slidingFitHalfWindow);

    double originalChi2(0.);
    CartesianPointVector currentPoints3D;
    this->ExtractResults(protoHitVector, originalChi2, currentPoints3D);

    try
    {
        const ThreeDSlidingFitResult originalSlidingFitResult(&currentPoints3D, layerWindow, layerPitch);
        const double originalChi2WrtFit(this->GetChi2WrtFit(originalSlidingFitResult, protoHitVector));
        double currentChi2(originalChi2 + originalChi2WrtFit);

        unsigned int nIterations(0);

        while (nIterations++ < m_nHitRefinementIterations)
        {
            ProtoHitVector newProtoHitVector(protoHitVector);
            const ThreeDSlidingFitResult newSlidingFitResult(&currentPoints3D, layerWindow, layerPitch);
            this->RefineHitPositions(newSlidingFitResult, newProtoHitVector);

            double newChi2(0.);
            CartesianPointVector newPoints3D;
            this->ExtractResults(newProtoHitVector, newChi2, newPoints3D);

            if (newChi2 > m_iterationMaxChi2Ratio * currentChi2)
                break;

            currentChi2 = newChi2;
            currentPoints3D = newPoints3D;
            protoHitVector = newProtoHitVector;
        }
    }
    catch (const StatusCodeException &)
    {
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::Project3DHit(const ProtoHit &hit, const HitType view, ProtoHit &projectedHit)
{
    projectedHit.SetPosition3D(
        LArGeometryHelper::ProjectPosition(this->GetPandora(), hit.GetPosition3D(), view),
        hit.GetChi2(),
        hit.IsInterpolated()
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::GetSetIntersection(RANSACHitVector &first, RANSACHitVector &second, RANSACHitVector &result)
{
    auto compareFunction = [] (const RANSACHit &a, const RANSACHit &b) -> bool {
        return a.GetProtoHit().GetPosition3D().GetX() < b.GetProtoHit().GetPosition3D().GetX();
    };

    std::sort(first.begin(), first.end(), compareFunction);
    std::sort(second.begin(), second.end(), compareFunction);

    std::set_intersection(
        first.begin(), first.end(),
        second.begin(), second.end(),
        std::inserter(result, result.begin()),
        compareFunction
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::ConsolidatedMethod(const ParticleFlowObject *const pPfo, ProtoHitVectorMap &allProtoHitVectors,
        ProtoHitVector &protoHitVector)
{
    if (allProtoHitVectors.size() == 0)
        return;

    // TODO: Drop logging.
    std::cout << "Starting consolidation method..." << std::endl;

    const float DISTANCE_THRESHOLD = 0.05; // TODO: Move to config option.
    const std::vector<HitType> views = {TPC_VIEW_U, TPC_VIEW_V, TPC_VIEW_W};

    std::map<HitType, RANSACHitVector> goodHits;

    const std::vector<std::string> toolsToAvoid = {"Tool0039", "Tool0043"}; // TODO: Config option?

    for (auto toolVectorPair : allProtoHitVectors)
    {
        if (toolVectorPair.second.size() == 0)
            continue;

        // TODO: Drop logging.
        std::cout << toolVectorPair.first << " contributed hits..." << std::endl;

        // INFO: Project every 3D hit into all 2D views, so how well they match
        // can be compared.
        for (const auto &hit : toolVectorPair.second)
        {
            const CaloHit* twoDHit = hit.GetParentCaloHit2D();

            for (HitType view : views)
            {
                ProtoHit hitForView(twoDHit);
                this->Project3DHit(hit, view, hitForView);

                const float disp = std::fabs(hitForView.GetPosition3D().GetX() - twoDHit->GetPositionVector().GetX());

                if (disp <= DISTANCE_THRESHOLD)
                {
                    auto avoidedIt = std::find(toolsToAvoid.begin(), toolsToAvoid.end(), toolVectorPair.first);
                    const bool goodTool = avoidedIt == toolsToAvoid.end();
                    goodHits[view].push_back(RANSACHit(hit, goodTool));
                }
            }
        }
    }

    RANSACHitVector UVconsistentHits;
    this->GetSetIntersection(goodHits[TPC_VIEW_V], goodHits[TPC_VIEW_U], UVconsistentHits);

    RANSACHitVector consistentHits;
    this->GetSetIntersection(goodHits[TPC_VIEW_W], UVconsistentHits, consistentHits);

    const float pitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    LArRANSACMethod ransacMethod(pitch, consistentHits);
    ransacMethod.Run(protoHitVector);

    ProtoHitVector consistentProtoHits;
    for (auto hit : consistentHits)
        consistentProtoHits.push_back(hit.GetProtoHit());

    // TODO: Drop all metric code.
    ransacMethod.m_allProtoHitsToPlot.push_back(std::make_pair("goodHits", consistentProtoHits));
    ransacMethod.m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_preInterpolation", protoHitVector));
    this->InterpolationMethod(pPfo, protoHitVector);
    ransacMethod.m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_preSmoothing", protoHitVector));
    this->IterativeTreatment(protoHitVector);
    ransacMethod.m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_chosen", protoHitVector));

    this->OutputDebugMetrics(pPfo, protoHitVector, allProtoHitVectors, ransacMethod.m_allProtoHitsToPlot);
}


//------------------------------------------------------------------------------------------------------------------------------------------

// TODO: Remove.
void ThreeDHitCreationAlgorithm::OutputDebugMetrics(
        const ParticleFlowObject *const pPfo,
        const ProtoHitVector &protoHitVector,
        const ProtoHitVectorMap &allProtoHitVectors,
        const std::vector<std::pair<std::string, ProtoHitVector>> &allProtoHitsToPlot
)
{
    bool printMetrics = false;
    bool dumpCSVs = true;

    if (dumpCSVs)
    {
        OutputCSVs(pPfo, allProtoHitVectors, allProtoHitsToPlot);
        return;
    }

    if (!printMetrics)
        return;

    const MCParticleList *pMCParticleList = nullptr;
    StatusCode mcReturn = PandoraContentApi::GetList(*this, m_mcParticleListName, pMCParticleList);

    CartesianPointVector pointVector;
    CaloHitVector twoDHits;
    CartesianPointVector pointVectorMC;

    for (const auto &nextPoint : protoHitVector)
    {
        pointVector.push_back(nextPoint.GetPosition3D());
        twoDHits.push_back(nextPoint.GetParentCaloHit2D());
    }

    const LArTPC *const pFirstLArTPC(this->GetPandora().GetGeometry()->GetLArTPCMap().begin()->second);
    metricParams params;

    params.layerPitch = pFirstLArTPC->GetWirePitchW();
    params.slidingFitWidth = m_slidingFitHalfWindow;

    threeDMetric metrics;
    this->initMetrics(metrics);

    if (mcReturn == STATUS_CODE_SUCCESS)
    {
        MCParticleList mcList(pMCParticleList->begin(), pMCParticleList->end());
        const MCParticle *const pMCParticle = LArMCParticleHelper::GetMainMCParticle(pPfo);
        const LArMCParticle *const pLArMCParticle(dynamic_cast<const LArMCParticle *>(pMCParticle));
        metrics.particleId = pMCParticle->GetParticleId();

        if (pLArMCParticle != NULL)
        {
            for (const auto &nextMCHit : pLArMCParticle->GetMCStepPositions())
                pointVectorMC.push_back(LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit));
        }
    }

    LArMetricHelper::GetThreeDMetrics(this->GetPandora(), pPfo, pointVector, twoDHits, metrics, params, pointVectorMC);

#ifdef MONITORING
    this->setupMetricsPlot();
    this->plotMetrics(pPfo, metrics);
    this->tearDownMetricsPlot(true);
#endif
}

//------------------------------------------------------------------------------------------------------------------------------------------

// TODO: Remove.
void ThreeDHitCreationAlgorithm::OutputCSVs(
        const ParticleFlowObject *const pPfo,
        const ProtoHitVectorMap &allProtoHitVectors,
        const std::vector<std::pair<std::string, ProtoHitVector>> &allProtoHitsToPlot
) const
{
    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    CaloHitVector twoDHits;
    for (ProtoHitVectorMap::value_type protoHitVectorPair : allProtoHitVectors)
    {
        for (const auto &hit : protoHitVectorPair.second)
        {
            const CaloHit* twoDHit = hit.GetParentCaloHit2D();

            if (std::find(twoDHits.begin(), twoDHits.end(), twoDHit) == twoDHits.end())
                twoDHits.push_back(twoDHit);
        }
    }

    while (true)
    {

        fileName = "/home/scratch/threeDHits/recoHits_" +
            std::to_string(fileNum) +
            ".csv";
        std::ifstream testFile(fileName.c_str());

        if (!testFile.good())
            break;

        testFile.close();
        ++fileNum;
    }

    std::ofstream csvFile;
    csvFile.open(fileName);

    csvFile << "X, Y, Z, ChiSquared, Interpolated, ToolName" << std::endl;
    for (auto &hitTwoD : twoDHits)
        csvFile << hitTwoD->GetPositionVector().GetX() << ","
            << hitTwoD->GetPositionVector().GetY() << ","
            << hitTwoD->GetPositionVector().GetZ() << ","
            << "0,0,2D" << std::endl;

    for (auto pair : allProtoHitVectors)
    {
        if (pair.second.size() == 0)
            continue;

        csvFile << "X, Y, Z, ChiSquared, Interpolated, ToolName" << std::endl;

        for (auto &hitThreeD : pair.second) {
            csvFile << hitThreeD.GetPosition3D().GetX() << ","
                << hitThreeD.GetPosition3D().GetY() << ","
                << hitThreeD.GetPosition3D().GetZ() << ","
                << hitThreeD.GetChi2() << ","
                << (hitThreeD.IsInterpolated() ? 1 : 0) << ","
                << pair.first
                << std::endl;
        }
    }

    if (allProtoHitsToPlot.size() > 0)
    {
        for (auto nameVectorPair : allProtoHitsToPlot)
        {
            if (nameVectorPair.second.size() == 0)
                continue;

            csvFile << "X, Y, Z, ChiSquared, Interpolated, ToolName" << std::endl;
            std::string outputName = nameVectorPair.first;

            for (auto &hitThreeD : nameVectorPair.second)
            {
                csvFile << hitThreeD.GetPosition3D().GetX() << ","
                    << hitThreeD.GetPosition3D().GetY() << ","
                    << hitThreeD.GetPosition3D().GetZ() << ","
                    << hitThreeD.GetChi2() << ","
                    << (hitThreeD.IsInterpolated() ? 1 : 0) << ","
                    << outputName
                    << std::endl;
            }
        }
    }

    const MCParticleList *pMCParticleList = nullptr;
    StatusCode mcReturn = PandoraContentApi::GetList(*this, m_mcParticleListName, pMCParticleList);

    if (mcReturn == STATUS_CODE_SUCCESS)
    {
        MCParticleList mcList(pMCParticleList->begin(), pMCParticleList->end());
        const MCParticle *const pMCParticle = LArMCParticleHelper::GetMainMCParticle(pPfo);
        const LArMCParticle *const pLArMCParticle(dynamic_cast<const LArMCParticle *>(pMCParticle));

        if (pLArMCParticle != NULL)
        {
            csvFile << "X, Y, Z, ChiSquared, Interpolated, ToolName" << std::endl;

            for (const auto &nextMCHit : pLArMCParticle->GetMCStepPositions())
            {
                CartesianVector mcHit = LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit);
                csvFile << mcHit.GetX() << ","
                    << mcHit.GetY() << ","
                    << mcHit.GetZ() << ","
                    << "0,0,mcHits"
                    << std::endl;
            }

            for (const auto &nextDaughter : pLArMCParticle->GetDaughterList())
            {
                const LArMCParticle *const daughterParticle(dynamic_cast<const LArMCParticle *>(nextDaughter));

                for (const auto &nextMCHit : daughterParticle->GetMCStepPositions())
                {
                    CartesianVector mcHit = LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit);
                    csvFile << mcHit.GetX() << ","
                        << mcHit.GetY() << ","
                        << mcHit.GetZ() << ","
                        << "0,0,mcHits"
                        << std::endl;
                }
            }

            for (const auto &nextParent : pLArMCParticle->GetParentList())
            {
                const LArMCParticle *const parentParticle(dynamic_cast<const LArMCParticle *>(nextParent));

                for (const auto &nextMCHit : parentParticle->GetMCStepPositions())
                {
                    CartesianVector mcHit = LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit);
                    csvFile << mcHit.GetX() << ","
                        << mcHit.GetY() << ","
                        << mcHit.GetZ() << ","
                        << "0,0,mcHits"
                        << std::endl;
                }
            }
        }
    }

    csvFile.close();
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::InterpolationMethod(const ParticleFlowObject *const pfo, ProtoHitVector &protoHitVector) const
{
    if (protoHitVector.empty())
        return;

    CaloHitVector remainingTwoDHits;
    this->SeparateTwoDHits(pfo, protoHitVector, remainingTwoDHits);

    if (remainingTwoDHits.empty())
        return;

    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(100); // TODO: Check if this should be the same or different.

    double originalChi2(0.);
    CartesianPointVector currentPoints3D;
    this->ExtractResults(protoHitVector, originalChi2, currentPoints3D);

    if (currentPoints3D.size() <= 1)
        return;

    const ThreeDSlidingFitResult slidingFitResult(&currentPoints3D, layerWindow, layerPitch);
    // CartesianVector fitDirection = slidingFitResult.GetGlobalMaxLayerDirection();

    const float sizeBefore = protoHitVector.size();

    for (const pandora::CaloHit* currentCaloHit : remainingTwoDHits)
    {
        const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(currentCaloHit);

        const float rL(slidingFitResult.GetLongitudinalDisplacement(pointPosition));

        CartesianVector projectedPosition(0.f, 0.f, 0.f);
        CartesianVector projectedDirection(0.f, 0.f, 0.f);
        const StatusCode positionStatusCode(slidingFitResult.GetGlobalFitPosition(rL, projectedPosition));
        const StatusCode directionStatusCode(slidingFitResult.GetGlobalFitDirection(rL, projectedDirection));

        if (positionStatusCode != STATUS_CODE_SUCCESS || directionStatusCode != STATUS_CODE_SUCCESS)
            continue;

        // TODO: Tune cut off.
        // Now we have a position for the interpolated hit, we need to make a
        // protoHit out of it.  This includes the calculation of a chi2 value,
        // for how good the interpolation is.
        // const float displacement = projectedPosition.GetCrossProduct(fitDirection).GetMagnitude();
        // const float otherDisp = projectedPosition.GetCrossProduct(projectedDirection).GetMagnitude();

        // if (otherDisp > 150)
        //     continue;

        ProtoHit interpolatedHit(currentCaloHit);

        const CartesianVector projectedHit = LArGeometryHelper::ProjectPosition(
                this->GetPandora(),
                projectedPosition,
                currentCaloHit->GetHitType()
        );
        const double distanceBetweenHitsSqrd = (
                (currentCaloHit->GetPositionVector() - projectedHit).GetMagnitudeSquared()
        );

        const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
        const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);
        const double interpolatedChi2 = (distanceBetweenHitsSqrd) / (sigma3DFit * sigma3DFit);

        interpolatedHit.SetPosition3D(projectedPosition, interpolatedChi2, true);
        interpolatedHit.AddTrajectorySample(
                TrajectorySample(projectedPosition, currentCaloHit->GetHitType(), sigmaUVW)
        );

        protoHitVector.push_back(interpolatedHit);
    }

    // ATTN: If we've interpolated at least 80% of this particle, don't use it.
    //
    // TODO: Swap to option?
    // TODO: This ideally would be earlier on, and wouldn't clear, but just drop the interpolated.
    const float numberOfInterpolatedHits = protoHitVector.size() - sizeBefore;
    if (numberOfInterpolatedHits >= (0.8 * protoHitVector.size()))
        protoHitVector.clear();
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::ExtractResults(const ProtoHitVector &protoHitVector, double &chi2, CartesianPointVector &pointVector) const
{
    chi2 = 0.;
    pointVector.clear();

    for (const ProtoHit &protoHit : protoHitVector)
    {
        chi2 += protoHit.GetChi2();
        pointVector.push_back(protoHit.GetPosition3D());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ThreeDHitCreationAlgorithm::GetChi2WrtFit(const ThreeDSlidingFitResult &slidingFitResult, const ProtoHitVector &protoHitVector) const
{
    const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
    const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);

    double chi2WrtFit(0.);

    for (const ProtoHit &protoHit : protoHitVector)
    {
        CartesianVector pointOnFit(0.f, 0.f, 0.f);
        const double rL(slidingFitResult.GetLongitudinalDisplacement(protoHit.GetPosition3D()));

        if (STATUS_CODE_SUCCESS != slidingFitResult.GetGlobalFitPosition(rL, pointOnFit))
            continue;

        const double uFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double vFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double wFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(pointOnFit.GetY(), pointOnFit.GetZ()));

        const double outputU(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ()));
        const double outputV(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ()));
        const double outputW(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ()));

        const double deltaUFit(uFit - outputU), deltaVFit(vFit - outputV), deltaWFit(wFit - outputW);
        chi2WrtFit += ((deltaUFit * deltaUFit) / (sigma3DFit * sigma3DFit)) + ((deltaVFit * deltaVFit) / (sigma3DFit * sigma3DFit)) + ((deltaWFit * deltaWFit) / (sigma3DFit * sigma3DFit));
    }

    return chi2WrtFit;
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ThreeDHitCreationAlgorithm::GetHitMovementChi2(const ProtoHitVector &protoHitVector) const
{
    const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
    double hitMovementChi2(0.);

    for (const ProtoHit &protoHit : protoHitVector)
    {
        const CaloHit *const pCaloHit2D(protoHit.GetParentCaloHit2D());
        const HitType hitType(pCaloHit2D->GetHitType());

        const CartesianVector projectedPosition(LArGeometryHelper::ProjectPosition(this->GetPandora(), protoHit.GetPosition3D(), hitType));
        const double delta(static_cast<double>(pCaloHit2D->GetPositionVector().GetZ() - projectedPosition.GetZ()));

        hitMovementChi2 += (delta * delta) / (sigmaUVW * sigmaUVW);
    }

    return hitMovementChi2;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::RefineHitPositions(const ThreeDSlidingFitResult &slidingFitResult, ProtoHitVector &protoHitVector) const
{
    const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
    const double sigmaFit(sigmaUVW); // ATTN sigmaFit and sigmaHit here should agree with treatment in HitCreation tools
    const double sigmaHit(sigmaUVW);
    const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);

    for (ProtoHit &protoHit : protoHitVector)
    {
        CartesianVector pointOnFit(0.f, 0.f, 0.f);
        const double rL(slidingFitResult.GetLongitudinalDisplacement(protoHit.GetPosition3D()));

        if (STATUS_CODE_SUCCESS != slidingFitResult.GetGlobalFitPosition(rL, pointOnFit))
            continue;

        const CaloHit *const pCaloHit2D(protoHit.GetParentCaloHit2D());
        const HitType hitType(pCaloHit2D->GetHitType());

        const double uFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double vFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double wFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(pointOnFit.GetY(), pointOnFit.GetZ()));

        const double sigmaU((TPC_VIEW_U == hitType) ? sigmaHit : sigmaFit);
        const double sigmaV((TPC_VIEW_V == hitType) ? sigmaHit : sigmaFit);
        const double sigmaW((TPC_VIEW_W == hitType) ? sigmaHit : sigmaFit);

        CartesianVector position3D(0.f, 0.f, 0.f);
        double chi2(std::numeric_limits<double>::max());
        double u(std::numeric_limits<double>::max()), v(std::numeric_limits<double>::max()), w(std::numeric_limits<double>::max());

        if (protoHit.GetNTrajectorySamples() == 2)
        {
            u = (TPC_VIEW_U == hitType) ? pCaloHit2D->GetPositionVector().GetZ() : (TPC_VIEW_U == protoHit.GetFirstTrajectorySample().GetHitType()) ? protoHit.GetFirstTrajectorySample().GetPosition().GetZ() : protoHit.GetLastTrajectorySample().GetPosition().GetZ();
            v = (TPC_VIEW_V == hitType) ? pCaloHit2D->GetPositionVector().GetZ() : (TPC_VIEW_V == protoHit.GetFirstTrajectorySample().GetHitType()) ? protoHit.GetFirstTrajectorySample().GetPosition().GetZ() : protoHit.GetLastTrajectorySample().GetPosition().GetZ();
            w = (TPC_VIEW_W == hitType) ? pCaloHit2D->GetPositionVector().GetZ() : (TPC_VIEW_W == protoHit.GetFirstTrajectorySample().GetHitType()) ? protoHit.GetFirstTrajectorySample().GetPosition().GetZ() : protoHit.GetLastTrajectorySample().GetPosition().GetZ();
        }
        else if (protoHit.GetNTrajectorySamples() == 1)
        {
            u = PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ());
            v = PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ());
            w = PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ());
        }
        else
        {
            // std::cout << "ThreeDHitCreationAlgorithm::IterativeTreatment - Unexpected number of trajectory samples" << std::endl;
            // throw StatusCodeException(STATUS_CODE_FAILURE);
            continue; // TODO: Fix this, check why it happens and if it is specific to the changes I've made.
        }

        double bestY(std::numeric_limits<double>::max()), bestZ(std::numeric_limits<double>::max());
        PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->GetMinChiSquaredYZ(u, v, w, sigmaU, sigmaV, sigmaW, uFit, vFit, wFit, sigma3DFit, bestY, bestZ, chi2);
        position3D.SetValues(protoHit.GetPosition3D().GetX(), static_cast<float>(bestY), static_cast<float>(bestZ));

        protoHit.SetPosition3D(position3D, chi2, protoHit.IsInterpolated());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::CreateThreeDHits(const ProtoHitVector &protoHitVector, CaloHitList &newThreeDHits) const
{
    for (const ProtoHit &protoHit : protoHitVector)
    {
        const CaloHit *pCaloHit3D(nullptr);
        this->CreateThreeDHit(protoHit, pCaloHit3D);

        if (!pCaloHit3D)
            throw StatusCodeException(STATUS_CODE_FAILURE);

        newThreeDHits.push_back(pCaloHit3D);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::CreateThreeDHit(const ProtoHit &protoHit, const CaloHit *&pCaloHit3D) const
{
    if (!this->CheckThreeDHit(protoHit))
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

    PandoraContentApi::CaloHit::Parameters parameters;
    parameters.m_positionVector = protoHit.GetPosition3D();
    parameters.m_hitType = TPC_3D;

    const CaloHit *const pCaloHit2D(protoHit.GetParentCaloHit2D());
    parameters.m_pParentAddress = static_cast<const void*>(pCaloHit2D);

    // TODO Check these parameters, especially new cell dimensions
    parameters.m_cellThickness = pCaloHit2D->GetCellThickness();
    parameters.m_cellGeometry = RECTANGULAR;
    parameters.m_cellSize0 = pCaloHit2D->GetCellLengthScale();
    parameters.m_cellSize1 = pCaloHit2D->GetCellLengthScale();
    parameters.m_cellNormalVector = pCaloHit2D->GetCellNormalVector();
    parameters.m_expectedDirection = pCaloHit2D->GetExpectedDirection();
    parameters.m_nCellRadiationLengths = pCaloHit2D->GetNCellRadiationLengths();
    parameters.m_nCellInteractionLengths = pCaloHit2D->GetNCellInteractionLengths();
    parameters.m_time = pCaloHit2D->GetTime();
    parameters.m_inputEnergy = pCaloHit2D->GetInputEnergy();
    parameters.m_mipEquivalentEnergy = pCaloHit2D->GetMipEquivalentEnergy();
    parameters.m_electromagneticEnergy = pCaloHit2D->GetElectromagneticEnergy();
    parameters.m_hadronicEnergy = pCaloHit2D->GetHadronicEnergy();
    parameters.m_isDigital = pCaloHit2D->IsDigital();
    parameters.m_hitRegion = pCaloHit2D->GetHitRegion();
    parameters.m_layer = pCaloHit2D->GetLayer();
    parameters.m_isInOuterSamplingLayer = pCaloHit2D->IsInOuterSamplingLayer();
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CaloHit::Create(*this, parameters, pCaloHit3D));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDHitCreationAlgorithm::CheckThreeDHit(const ProtoHit &protoHit) const
{
    try
    {
        // Check that corresponding pseudo layer is within range - TODO use full LArTPC geometry here
        (void) PandoraContentApi::GetPlugins(*this)->GetPseudoLayerPlugin()->GetPseudoLayer(protoHit.GetPosition3D());
    }
    catch (StatusCodeException &)
    {
        return false;
    }

    // TODO Check against detector geometry
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::AddThreeDHitsToPfo(const ParticleFlowObject *const pPfo, const CaloHitList &caloHitList) const
{
    if (caloHitList.empty())
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    ClusterList threeDClusterList;
    LArPfoHelper::GetThreeDClusterList(pPfo, threeDClusterList);

    if (!threeDClusterList.empty())
        throw StatusCodeException(STATUS_CODE_FAILURE);

    const ClusterList *pClusterList(nullptr); std::string clusterListName;
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pClusterList, clusterListName));

    PandoraContentApi::Cluster::Parameters parameters;
    parameters.m_caloHitList.insert(parameters.m_caloHitList.end(), caloHitList.begin(), caloHitList.end());

    const Cluster *pCluster3D(nullptr);
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, parameters, pCluster3D));

    if (!pCluster3D || !pClusterList || pClusterList->empty())
        throw StatusCodeException(STATUS_CODE_FAILURE);

    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_outputClusterListName));
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToPfo(*this, pPfo, pCluster3D));
}

//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

const CartesianVector &ThreeDHitCreationAlgorithm::ProtoHit::GetPosition3D() const
{
    if (!m_isPositionSet)
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_position3D;
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ThreeDHitCreationAlgorithm::ProtoHit::GetChi2() const
{
    if (!m_isPositionSet)
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_chi2;
}

//------------------------------------------------------------------------------------------------------------------------------------------

const ThreeDHitCreationAlgorithm::TrajectorySample &ThreeDHitCreationAlgorithm::ProtoHit::GetFirstTrajectorySample() const
{
    if (m_trajectorySampleVector.empty())
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_trajectorySampleVector.front();
}

//------------------------------------------------------------------------------------------------------------------------------------------

const ThreeDHitCreationAlgorithm::TrajectorySample &ThreeDHitCreationAlgorithm::ProtoHit::GetLastTrajectorySample() const
{
    if (m_trajectorySampleVector.size() < 2)
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_trajectorySampleVector.back();
}

//------------------------------------------------------------------------------------------------------------------------------------------

// TODO: Remove.
void ThreeDHitCreationAlgorithm::initMetrics(threeDMetric &metricStruct) {
    // Set everything to -999, so we know it failed.
    metricStruct.particleId = -999;
    metricStruct.acosDotProductAverage = -999;
    metricStruct.trackDisplacementAverageMC = -999;
    metricStruct.distanceToFitAverage = -999;
    metricStruct.numberOf3DHits = -999;
    metricStruct.lengthOfTrack = -999;
    metricStruct.numberOfErrors = -999;

    metricStruct.recoUDisplacement = {-999};
    metricStruct.recoVDisplacement = {-999};
    metricStruct.recoWDisplacement = {-999};
    metricStruct.mcUDisplacement = {-999};
    metricStruct.mcVDisplacement = {-999};
    metricStruct.mcWDisplacement = {-999};
}

// TODO: Remove all.
#ifdef MONITORING
//------------------------------------------------------------------------------------------------------------------------------------------
void ThreeDHitCreationAlgorithm::setupMetricsPlot()
{
    // Find a file name by just picking a file name
    // until an unused one is found.
    int fileNum = 0;

    while (true)
    {

        m_metricFileName = "/home/scratch/threeDMetricOutput/threeDTrackEff_" +
            std::to_string(fileNum) +
            ".root";
        std::ifstream testFile(m_metricFileName.c_str());

        if (!testFile.good())
            break;

        testFile.close();
        ++fileNum;
    }

    // Make an output folder if needed.
    mkdir("/home/scratch/threeDMetricOutput", 0775);

    PANDORA_MONITORING_API(Create(this->GetPandora()));
}

//------------------------------------------------------------------------------------------------------------------------------------------
void ThreeDHitCreationAlgorithm::tearDownMetricsPlot(bool saveTree)
{
    if (saveTree)
        PANDORA_MONITORING_API(SaveTree(this->GetPandora(), m_metricTreeName.c_str(), m_metricFileName.c_str(), "RECREATE"));

    PANDORA_MONITORING_API(Delete(this->GetPandora()));
}
//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::plotMetrics(
        const ParticleFlowObject *const pInputPfo,
        threeDMetric &metricStruct
) {
    std::cout << "********** Did not bail out! Metrics will run." << std::endl;

    // Get the 2D clusters for this pfo.
    ClusterList clusterList;
    LArPfoHelper::GetTwoDClusterList(pInputPfo, clusterList);

    double convertedRatio =  0.0;
    double totalNumberOf2DHits = 0.0;
    double trackWasReconstructed  = 0.0;
    double reconstructionState = -999.0;

    for (auto cluster : clusterList)
        totalNumberOf2DHits += cluster->GetNCaloHits();

    // Set the converted ratio.
    // This is going to be between 0 and 1, or -999 in the case of bad reco.
    if (metricStruct.numberOf3DHits == -999)
    {
        convertedRatio = -999;
    }
    else if (metricStruct.numberOf3DHits != 0)
    {
        convertedRatio = metricStruct.numberOf3DHits / totalNumberOf2DHits;
    }
    else
    {
        convertedRatio = 0.0;
    }

    switch(metricStruct.valuesHaveBeenSet)
    {
        case errorCases::SUCCESSFULLY_SET:
            trackWasReconstructed = 1.0;
            break;
        case errorCases::ERROR:
        case errorCases::TRACK_BUILDING_ERROR:
        case errorCases::NO_VERTEX_ERROR:
            trackWasReconstructed = 0.0;
            break;
        case errorCases::NON_NEUTRINO:
        case errorCases::NON_FINAL_STATE:
        case errorCases::NON_TRACK:
        case errorCases::NOT_SET:
            trackWasReconstructed = -999;
            break;
    }

    std::cout << "Number of 2D Hits: " << totalNumberOf2DHits << std::endl;
    std::cout << "Number of 3D Hits: " << metricStruct.numberOf3DHits << std::endl;
    std::cout << "Ratio: " << convertedRatio << std::endl;

    switch(metricStruct.valuesHaveBeenSet)
    {
        case errorCases::NOT_SET:
            reconstructionState = 0;
            break;
        case errorCases::ERROR:
            reconstructionState = 1;
            break;
        case errorCases::SUCCESSFULLY_SET:
            reconstructionState = 2;
            break;
        case errorCases::NON_NEUTRINO:
            reconstructionState = 3;
            break;
        case errorCases::NON_FINAL_STATE:
            reconstructionState = 4;
            break;
        case errorCases::NON_TRACK:
            reconstructionState = 5;
            break;
        case errorCases::TRACK_BUILDING_ERROR:
            reconstructionState = 6;
            break;
        case errorCases::NO_VERTEX_ERROR:
            reconstructionState = 7;
            break;
        default:
            reconstructionState = -999;
            break;
    }

    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "particleId", metricStruct.particleId));

    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "acosDotProductAverage", metricStruct.acosDotProductAverage));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "sqdTrackDisplacementAverageMC", metricStruct.trackDisplacementAverageMC));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "distanceToFitAverage", metricStruct.distanceToFitAverage));

    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "numberOf3DHits", metricStruct.numberOf3DHits));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "numberOf2DHits", totalNumberOf2DHits));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "ratioOf3Dto2D", convertedRatio));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "numberOfErrors", metricStruct.numberOfErrors));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "lengthOfTrack", metricStruct.lengthOfTrack));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "trackWasReconstructed", trackWasReconstructed));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "reconstructionState", reconstructionState));

    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "recoUDisplacement", metricStruct.recoUDisplacement));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "recoVDisplacement", metricStruct.recoVDisplacement));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "recoWDisplacement", metricStruct.recoWDisplacement));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "mcUDisplacement", metricStruct.mcUDisplacement));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "mcVDisplacement", metricStruct.mcVDisplacement));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), m_metricTreeName.c_str(), "mcWDisplacement", metricStruct.mcWDisplacement));

    PANDORA_MONITORING_API(FillTree(this->GetPandora(), m_metricTreeName.c_str()));
    std::cout << "**********" << std::endl;
}
#endif

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeDHitCreationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    AlgorithmToolVector algorithmToolVector;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ProcessAlgorithmToolList(*this, xmlHandle,
        "HitCreationTools", algorithmToolVector));

    for (AlgorithmToolVector::const_iterator iter = algorithmToolVector.begin(), iterEnd = algorithmToolVector.end(); iter != iterEnd; ++iter)
    {
        HitCreationBaseTool *const pHitCreationTool(dynamic_cast<HitCreationBaseTool*>(*iter));

        if (!pHitCreationTool)
            return STATUS_CODE_INVALID_PARAMETER;

        m_algorithmToolVector.push_back(pHitCreationTool);
    }

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputPfoListName", m_inputPfoListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputCaloHitListName", m_outputCaloHitListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputClusterListName", m_outputClusterListName));

    // TODO: Remove.
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MCParticleListName", m_mcParticleListName));

    // TODO: Remove.
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MetricTreeFileName", m_metricFileName));

    // TODO: Remove.
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MetricTreeName", m_metricTreeName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterateTrackHits", m_iterateTrackHits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterateShowerHits", m_iterateShowerHits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UseRANSAC", m_useRANSACMethod));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "InterpolationCut", m_interpolationCutOff));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SlidingFitHalfWindow", m_slidingFitHalfWindow));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "NHitRefinementIterations", m_nHitRefinementIterations));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "Sigma3DFitMultiplier", m_sigma3DFitMultiplier));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterationMaxChi2Ratio", m_iterationMaxChi2Ratio));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
