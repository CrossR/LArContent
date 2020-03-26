/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.cc
 *
 *  @brief  Implementation of the three dimensional hit creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArFileHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArObjectHelper.h"
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include "larpandoracontent/LArObjects/LArMCParticle.h"
#include "larpandoracontent/LArObjects/LArAdaBoostDecisionTree.h"
#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"

#include "larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

#include <algorithm>
#include <fstream>
#include <sys/stat.h>

#include "larpandoracontent/LArUtility/RANSAC/PlaneModel.h"
#include "larpandoracontent/LArUtility/RANSAC/RANSAC.h"

#ifdef MONITORING
#include "PandoraMonitoringApi.h"
#endif

using namespace pandora;

namespace lar_content
{

ThreeDHitCreationAlgorithm::ThreeDHitCreationAlgorithm() :
    m_trackMVAFileName(""),
    m_metricFileName(""),
    m_metricTreeName("threeDTrackTree"),
    m_iterateTrackHits(true),
    m_iterateShowerHits(false),
    m_useInterpolation(false),
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

            try
            {
                pHitCreationTool->Run(this, pPfo, remainingTwoDHits, protoHitVector);

                if (m_useInterpolation && LArPfoHelper::IsTrack(pPfo))
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

                // Insert an entry for cases that failed, to help with training.
                if (m_useInterpolation && LArPfoHelper::IsTrack(pPfo))
                {
                    allProtoHitVectors.insert(ProtoHitVectorMap::value_type(pHitCreationTool->GetInstanceName(), protoHitVector));
                    protoHitVector.clear();
                }

                continue;
            }
        }

        if (numberOfFailedAlgorithms == m_algorithmToolVector.size())
            throw StatusCodeException(STATUS_CODE_FAILURE);

        bool shouldUseIterativeTreatment = (
                (m_iterateTrackHits && LArPfoHelper::IsTrack(pPfo)) ||
                (m_iterateShowerHits && LArPfoHelper::IsShower(pPfo))
        );

        if (shouldUseIterativeTreatment && !m_useInterpolation)
        {
            this->IterativeTreatment(protoHitVector);
        }

        if (m_useInterpolation && LArPfoHelper::IsTrack(pPfo))
        {
            this->ConsolidatedMethod(pPfo, allProtoHitVectors, protoHitVector);
            allProtoHitVectors.clear();
        }

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

void ThreeDHitCreationAlgorithm::GetSetIntersection(ProtoHitVector &first, ProtoHitVector &second, ProtoHitVector &result)
{
    auto compareFunction = [] (const ProtoHit &a, const ProtoHit &b) -> bool {
        return a.GetPosition3D().GetX() < b.GetPosition3D().GetX();
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

    std::cout << "Starting consolidation method..." << std::endl;

    const float DISTANCE_THRESHOLD = 0.05; // TODO: Move to config option.
    const std::vector<HitType> views = {TPC_VIEW_U, TPC_VIEW_V, TPC_VIEW_W};

    std::map<HitType, ProtoHitVector> goodHits;

    for (ProtoHitVectorMap::value_type protoHitVectorPair : allProtoHitVectors)
    {
        if (protoHitVectorPair.second.size() == 0)
            continue;

        std::cout << protoHitVectorPair.first << " contributed hits..." << std::endl;

        for (const auto &hit : protoHitVectorPair.second)
        {
            const CaloHit* twoDHit = hit.GetParentCaloHit2D();

            for (HitType view : views)
            {
                ProtoHit hitForView(twoDHit);
                this->Project3DHit(hit, view, hitForView);

                bool goodHit = std::fabs(hitForView.GetPosition3D().GetX() - twoDHit->GetPositionVector().GetX()) <= DISTANCE_THRESHOLD;

                if (goodHit)
                    goodHits[view].push_back(hit);
            }
        }
    }

    ProtoHitVector UVconsistentHits;
    this->GetSetIntersection(goodHits[TPC_VIEW_V], goodHits[TPC_VIEW_U], UVconsistentHits);

    ProtoHitVector consistentHits;
    this->GetSetIntersection(goodHits[TPC_VIEW_W], UVconsistentHits, consistentHits);

    ParameterVector candidatePoints;
    std::vector<std::pair<std::string, ProtoHitVector>> allProtoHitsToPlot;
    allProtoHitsToPlot.push_back(std::make_pair("goodHits", consistentHits));

    // TODO: Possibly want to sample this set for the initial RANSAC fit, just to speed it up.
    //       Having a cut off over X thousand hits perhaps?
    for (auto hit : consistentHits)
        candidatePoints.push_back(std::make_shared<Point3D>(hit));

    if (consistentHits.size() < 3)
        return; // TODO: Check if/what to return here.

    RANSAC<PlaneModel, 3> estimator;
    const float RANSAC_THRESHOLD = 2.5;
    const int RANSAC_ITERS = 100; // TODO: Should either be dynamic, or a config option.
    estimator.Initialize(RANSAC_THRESHOLD, RANSAC_ITERS);
    std::cout << "Starting " << RANSAC_ITERS << " RANSAC iterations..." << std::endl;
    estimator.Estimate(candidatePoints);
    std::cout << "Best RANSAC size after initial run: " << estimator.GetBestInliers().size() << std::endl;
    std::cout << "Second RANSAC size after initial run: " << estimator.GetSecondBestInliers().size() << std::endl;

    std::vector<std::pair<std::string, ParameterVector>> parameterVectors;
    parameterVectors.push_back(std::make_pair("bestInliers", estimator.GetBestInliers()));
    parameterVectors.push_back(std::make_pair("secondBestInliers", estimator.GetSecondBestInliers()));

    // TODO: This gets the job done...but actually do it properly. Either pass them over, or do it nicer.
    /*******************************************************************************************************/
    std::map<const CaloHit*, std::pair<ProtoHit, float>> inlyingHitMap;
    for (auto inlier : estimator.GetBestInliers())
        this->AddToHitMap((*std::dynamic_pointer_cast<Point3D>(inlier)).m_ProtoHit, inlyingHitMap, 0.0);

    std::map<const CaloHit*, std::pair<ProtoHit, float>> inlyingHitMap2;
    for (auto inlier : estimator.GetSecondBestInliers())
        this->AddToHitMap((*std::dynamic_pointer_cast<Point3D>(inlier)).m_ProtoHit, inlyingHitMap2, 0.0);

    // Get the non-inlying hits, since they are what we want to run over next.
    // Set this up before iterating, to update it each time to remove stuff.
    ProtoHitVector nextHits;
    for (auto hit : consistentHits)
        if ((inlyingHitMap.count(hit.GetParentCaloHit2D()) == 0))
            if ((inlyingHitMap2.count(hit.GetParentCaloHit2D()) == 0))
                nextHits.push_back(hit);
    /*******************************************************************************************************/

    ProtoHitVector primaryResult;
    ProtoHitVector secondaryResult;
    std::cout << "Hits going into best run: " << nextHits.size() << std::endl;
    int primaryModelCount = this->RunOverRANSACOutput(
            pPfo, *estimator.GetBestModel(), estimator.GetBestInliers(), nextHits, primaryResult,
            allProtoHitsToPlot, "best"
    );
    std::cout << "Primary Result size: " << primaryResult.size() << std::endl;
    std::cout << "Hits going into second run: " << nextHits.size() << std::endl;
    int secondModelCount = this->RunOverRANSACOutput(
            pPfo, *estimator.GetSecondBestModel(), estimator.GetSecondBestInliers(), nextHits, secondaryResult,
            allProtoHitsToPlot, "second"
    );
    std::cout << "Secondary Result size: " << secondaryResult.size() << std::endl;

    int primaryTotal = estimator.GetBestInliers().size() + primaryModelCount;
    int secondaryTotal = estimator.GetSecondBestInliers().size() + secondModelCount;
    std::cout << primaryTotal << " vs " << secondaryTotal << std::endl;
    protoHitVector = primaryTotal > secondaryTotal ? primaryResult : secondaryResult;

    this->OutputDebugMetrics(pPfo, allProtoHitVectors, allProtoHitsToPlot, parameterVectors);
}

int ThreeDHitCreationAlgorithm::RunOverRANSACOutput(const ParticleFlowObject *const pPfo,
        PlaneModel &currentModel, ParameterVector &currentInliers, ProtoHitVector &hitsToUse,
        ProtoHitVector &protoHitVector,
        std::vector<std::pair<std::string, ProtoHitVector>> &allProtoHitsToPlot, std::string name
)
{
    // TODO: Check all the returns in here. The worst case should be protoHitVector = currentInliers.

    if (currentInliers.size() < 1 || hitsToUse.size() < 1)
        return 0;

    std::map<const CaloHit*, std::pair<ProtoHit, float>> inlyingHitMap;
    const float RANSAC_THRESHOLD = 2.5; // TODO: Consolidate to config option.

    // Now, use the RANSAC model to seed a sliding linear fit to finish
    // off the rest of the hits. This is hopefully more robust in that
    // the hits should all be consistent with the chosen hits, rather
    // than doing subsequent RANSAC runs and hoping the results are
    // consistent.
    for (auto inlier : currentInliers)
    {
        auto hit = std::dynamic_pointer_cast<Point3D>(inlier);

        if (hit == nullptr)
            throw std::runtime_error("Inlying hit was not of type Point3D");

        ProtoHit protoHit = (*hit).m_ProtoHit;
        this->AddToHitMap(protoHit, inlyingHitMap, 0.0); // TODO: Setting this to 0 makes the RANSAC hits permanent. Is that what we want?
    }

    // Get the non-inlying hits, since they are what we want to run over next.
    // Set this up before iterating, to update it each time to remove stuff.
    ProtoHitVector nextHits;
    for (auto hit : hitsToUse)
        if ((inlyingHitMap.count(hit.GetParentCaloHit2D()) == 0))
            nextHits.push_back(hit);

    CartesianVector fitOrigin = currentModel.GetOrigin();
    CartesianVector fitDirection = currentModel.GetDirection();
    auto sortByModelDisplacement = [&fitOrigin, &fitDirection](ProtoHit a, ProtoHit b) {
        float displacementA = (a.GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        float displacementB = (b.GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        return displacementA < displacementB;
    };

    // Get the hits we will be using for the initial sliding fit.
    ProtoHitVector smoothedHits;
    for (auto const& caloProtoPair : inlyingHitMap)
        smoothedHits.push_back(caloProtoPair.second.first);

    if (smoothedHits.size() < 3)
        return 0; // TODO: Work out what to do here, rather than just returning, since we have some stuff to do to the inliers.

    this->IterativeTreatment(smoothedHits);

    // TODO: If RANSAC was good enough (i.e. it got everything) skip this next bit.
    std::list<ProtoHit> currentPoints3D;

    for (auto hit : smoothedHits)
        currentPoints3D.push_back(hit);

    currentPoints3D.sort(sortByModelDisplacement);

    std::cout << "Before iterations " << inlyingHitMap.size() << std::endl;

    const int FIT_ITERATIONS = 1000; // TODO: Config?
    const int HITS_TO_KEEP = 80; // TODO: Config?

    ProtoHitVector hitsToUseForFit;
    if (currentPoints3D.size() <= HITS_TO_KEEP)
    {
        hitsToUseForFit = std::vector<ProtoHit>(currentPoints3D.begin(), currentPoints3D.end());
        currentPoints3D.clear();
    }
    else
    {
        auto it = currentPoints3D.begin();
        while(it != currentPoints3D.end())
        {
            hitsToUseForFit.push_back(*it);
            it = currentPoints3D.erase(it);

            if (hitsToUseForFit.size() >= HITS_TO_KEEP)
                break;
        }
    }

    // Run fit from start to end, including added hits.
    bool finishingUp = false;
    int coherentHitCount = 0;
    for (unsigned int iter = 0; iter < FIT_ITERATIONS; ++iter)
    {
        std::vector<std::pair<ProtoHit, float>> hitsToAddToFit;
        const float FIT_THRESHOLD = 20;
        int hitsAdded = 0;

        for (float fits = 1; fits < 4.0; ++fits)
        {
            ThreeDHitCreationAlgorithm::ExtendFit(
                nextHits, hitsToUseForFit, hitsToAddToFit,
                (FIT_THRESHOLD * fits), (RANSAC_THRESHOLD * fits),
                allProtoHitsToPlot, iter, name
            );
            hitsAdded = hitsToAddToFit.size();

            if (hitsAdded > 0)
            {
                std::cout << "  Final Settings: " << (FIT_THRESHOLD * fits) << " , " << (RANSAC_THRESHOLD * fits) << std::endl;
                break;
            }
        }

        // Add the best hits.
        // Update the hits that are used for the fits.
        for (auto hitDispPair : hitsToAddToFit)
        {
            bool hitAdded = this->AddToHitMap(hitDispPair.first, inlyingHitMap, hitDispPair.second);

            // Always increment this, so we represent every hit from every tool.
            ++coherentHitCount;

            if (hitAdded)
                hitsToUseForFit.push_back(hitDispPair.first);
        }

        // If we added no hits at the end, we should stop.
        // If we added no hits, but are looking inside the fit, rebuild the fit over the next N points.
        // If we added some hits, just drop enough hits to keep the fit small.
        // If we added some hits and the fit is the right size, carry on.
        if (hitsAdded == 0 && currentPoints3D.size() == 0)
            break;
        else if (hitsAdded < 5 && finishingUp)
            break;
        else if (hitsAdded <= 2 && currentPoints3D.size() != 0)
        {
            hitsToUseForFit.clear();
            auto it = currentPoints3D.begin();
            while(it != currentPoints3D.end())
            {
                hitsToUseForFit.push_back(*it);
                it = currentPoints3D.erase(it);

                if (hitsToUseForFit.size() >= HITS_TO_KEEP)
                    break;
            }
        }
        else if (hitsAdded > 0)
        {
            auto it = hitsToUseForFit.begin();
            while (hitsToUseForFit.size() > HITS_TO_KEEP)
                it = hitsToUseForFit.erase(it);
        }

        if (hitsAdded < 5 && currentPoints3D.size() == 0)
            finishingUp = true;

        std::cout << iter << ") Added: " << hitsAdded
                  << ", Left: " << currentPoints3D.size()
                  << ", Finishing: " << finishingUp << std::endl;
    }

    for (auto const& caloProtoPair : inlyingHitMap)
        protoHitVector.push_back(caloProtoPair.second.first);

    allProtoHitsToPlot.push_back(std::make_pair("preIterativeHits_" + name, protoHitVector));
    this->IterativeTreatment(protoHitVector);
    this->InterpolationMethod(pPfo, protoHitVector);
    allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_" + name, protoHitVector));

    std::cout << "At the end of extending, the protoHitVector was of size: " << protoHitVector.size() << std::endl;

    return coherentHitCount;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDHitCreationAlgorithm::AddToHitMap(
    ProtoHit hit,
    std::map<const CaloHit*, std::pair<ProtoHit, float>> &inlyingHitMap,
    float metricValue
)
{
    const CaloHit* twoDHit =  hit.GetParentCaloHit2D();

    if (inlyingHitMap.count(twoDHit) == 0)
    {
        inlyingHitMap[twoDHit] = std::make_pair(hit, metricValue);
        return true;
    }
    else
    {
        const float bestMetric = inlyingHitMap[twoDHit].second;
        if (metricValue < bestMetric)
        {
            inlyingHitMap[twoDHit] = std::make_pair(hit, metricValue);
            return true;
        }
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::ExtendFit(
    ProtoHitVector &hitsToTestAgainst, // TODO: List?
    ProtoHitVector &hitsToUseForFit,
    std::vector<std::pair<ProtoHit, float>> &hitsToAddToFit,
    const float distanceToEndThreshold,
    const float distanceToFitThreshold,
    std::vector<std::pair<std::string, ProtoHitVector>> &allProtoHitsToPlot,
    int iter, std::string name
)
{
    if (hitsToUseForFit.size() == 0)
        return;

    CartesianPointVector fitPoints;
    for (auto protoHit : hitsToUseForFit)
        fitPoints.push_back(protoHit.GetPosition3D());

    // TODO: Remove. Used for debugging.
    /*****************************************/
    ProtoHitVector hitsUsedInInitialFit;
    ProtoHitVector hitsToCheckForFit;
    ProtoHitVector hitsAddedToFit;
    ProtoHitVector hitsAddedToFitDisp;
    ProtoHitVector hitsCloseToFit;

    for (auto protoHit : hitsToUseForFit)
    {
        ProtoHit newHit(protoHit.GetParentCaloHit2D());
        newHit.SetPosition3D(protoHit.GetPosition3D(), iter, 0);
        hitsUsedInInitialFit.push_back(newHit);
    }

    for (auto protoHit : hitsToTestAgainst)
    {
        ProtoHit newHit(protoHit.GetParentCaloHit2D());
        newHit.SetPosition3D(protoHit.GetPosition3D(), iter, 0);
        hitsToCheckForFit.push_back(newHit);
    }
    /*****************************************/

    // We've now got a sliding linear fit that should be based on the RANSAC fit.
    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(100); // TODO: May want this one to be different, since its for a different use.
    const ThreeDSlidingFitResult slidingFitResult(&fitPoints, layerWindow, layerPitch);

    CartesianVector fitDirection = slidingFitResult.GetGlobalMaxLayerDirection();
    CartesianVector fitEnd = slidingFitResult.GetGlobalMaxLayerPosition();

    int addedHits = 0;
    std::vector<std::vector<ProtoHit>::iterator> hitsToPotentiallyCheck;

    // Use this sliding linear fit to test out the upcoming hits and pull some in
    auto it = hitsToTestAgainst.begin();
    while (it != hitsToTestAgainst.end())
    {
        // Get the position relative to the fit for the point.
        auto hit = *it;
        const CartesianVector pointPosition = hit.GetPosition3D();
        float dispFromFitEnd = (pointPosition - fitEnd).GetDotProduct(fitDirection);

        // If its not near the end of the fit, lets leave it to a subsequent iteration.
        if (dispFromFitEnd < 0.0 || abs(dispFromFitEnd) > distanceToEndThreshold)
        {
            ++it;
            continue;
        }

        // TODO: Remove. Used for debugging.
        /*****************************************/
        ProtoHit distHit(hit.GetParentCaloHit2D());
        distHit.SetPosition3D(hit.GetPosition3D(), dispFromFitEnd, 0);
        hitsCloseToFit.push_back(distHit);
        /*****************************************/

        CartesianVector projectedPosition(0.f, 0.f, 0.f);
        CartesianVector projectedDirection(0.f, 0.f, 0.f);
        const float rL(slidingFitResult.GetLongitudinalDisplacement(pointPosition));
        const StatusCode positionStatusCode(slidingFitResult.GetGlobalFitPosition(rL, projectedPosition));
        const StatusCode directionStatusCode(slidingFitResult.GetGlobalFitDirection(rL, projectedDirection));

        // Hits that failed to project but are still good are stored.
        // If we fail to project lots of hits, come back to these hits.
        if (positionStatusCode != STATUS_CODE_SUCCESS || directionStatusCode != STATUS_CODE_SUCCESS)
        {
            // We only use this when no hits are added, so iterator should be consistent.
            hitsToPotentiallyCheck.push_back(it);

            ++it;
            continue;
        }

        const float displacement = (pointPosition - projectedPosition).GetCrossProduct(projectedDirection).GetMagnitude();

        // If its good enough, lets store it and then we can pick the best one out later on.
        // Otherwise, just ignore it for now.
        if (displacement > distanceToFitThreshold)
        {
            ++it;
            continue;
        }

        // Since we are adding this hit, remove it from the hitsToTestAgainst vector.
        it = hitsToTestAgainst.erase(it);

        // TODO: Remove. Used for debugging.
        /*****************************************/
        ProtoHit newHit(hit.GetParentCaloHit2D());
        newHit.SetPosition3D(hit.GetPosition3D(), iter, 0);
        hitsAddedToFit.push_back(newHit);

        ProtoHit dispHit(hit.GetParentCaloHit2D());
        dispHit.SetPosition3D(hit.GetPosition3D(), displacement, 0);
        hitsAddedToFitDisp.push_back(dispHit);
        /*****************************************/

        ++addedHits;
        hitsToAddToFit.push_back(std::make_pair(hit, displacement));
    }

    // TODO: Remove. Used for debugging.
    /*****************************************/
    // std::cout << "############################################################################" << std::endl;
    // std::cout << "We used " << hitsToTestAgainst.size() << " hits in iteration..." << iter << std::endl;
    // std::cout << "We added " << addedHits << " hits in iteration..." << iter << std::endl;
    // std::cout << "There was " << hitsToPotentiallyCheck.size() << " hits to maybe use in iteration " << iter << "..." << std::endl;
    /*****************************************/

    if (addedHits != 0)
    {
        // std::cout << "############################################################################" << std::endl;
        allProtoHitsToPlot.push_back(std::make_pair("hitsUsedInFit_"    + name + "_" + std::to_string(iter), hitsUsedInInitialFit));
        allProtoHitsToPlot.push_back(std::make_pair("hitsToBeTested_"   + name + "_" + std::to_string(iter), hitsToCheckForFit));
        allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFit_"   + name + "_" + std::to_string(iter), hitsAddedToFit));
        allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFitDisp_"   + name + "_" + std::to_string(iter), hitsAddedToFitDisp));
        allProtoHitsToPlot.push_back(std::make_pair("hitsCloseToFit_"   + name + "_" + std::to_string(iter), hitsCloseToFit));
        return;
    }

    float sumOfDisplacements = 0.0;

    // Now, lets instead just use the end of the fit to compare against.
    for (auto hitIterator : hitsToPotentiallyCheck)
    {
        auto hit = *hitIterator;
        CartesianVector pointPosition = hit.GetPosition3D();
        const float displacement = (pointPosition - fitEnd).GetCrossProduct(fitDirection).GetMagnitude();

        if (displacement > distanceToFitThreshold)
            continue;

        sumOfDisplacements += displacement;

        ++addedHits;
        hitsToAddToFit.push_back(std::make_pair(hit, displacement));
        hitsToTestAgainst.erase(hitIterator);

        // TODO: Remove. Used for debugging.
        /*****************************************/
        ProtoHit newHit(hit.GetParentCaloHit2D());
        newHit.SetPosition3D(hit.GetPosition3D(), iter, 0);
        hitsAddedToFit.push_back(newHit);

        ProtoHit dispHit(hit.GetParentCaloHit2D());
        dispHit.SetPosition3D(hit.GetPosition3D(), displacement, 0);
        hitsAddedToFitDisp.push_back(dispHit);
        /*****************************************/
    }

    /*****************************************/
    // std::cout << "We finally added " << addedHits << " hits in iteration " << iter << "..." << std::endl;
    // std::cout << "Avg displacement for fallback was " << sumOfDisplacements/float(addedHits) << " in iteration " << iter << "..." << std::endl;
    // std::cout << "############################################################################" << std::endl;
    allProtoHitsToPlot.push_back(std::make_pair("hitsUsedInFit_"    + name + "_" + std::to_string(iter), hitsUsedInInitialFit));
    allProtoHitsToPlot.push_back(std::make_pair("hitsToBeTested_"   + name + "_" + std::to_string(iter), hitsToCheckForFit));
    allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFit_"   + name + "_" + std::to_string(iter), hitsAddedToFit));
    allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFitDisp_"   + name + "_" + std::to_string(iter), hitsAddedToFitDisp));
    allProtoHitsToPlot.push_back(std::make_pair("hitsCloseToFit_"   + name + "_" + std::to_string(iter), hitsCloseToFit));
    /*****************************************/
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::OutputDebugMetrics(
        const ParticleFlowObject *const pPfo,
        const ProtoHitVectorMap &allProtoHitVectors,
        const std::vector<std::pair<std::string, ProtoHitVector>> &allProtoHitsToPlot,
        const std::vector<std::pair<std::string, ParameterVector>> &parameterVectors
)
{
    bool printMetrics = false;
    bool visualiseHits = false;
    bool dumpCSVs = true;

    if (dumpCSVs)
        OutputCSVs(pPfo, allProtoHitVectors, allProtoHitsToPlot, parameterVectors);

    std::vector<std::pair<std::string, threeDMetric>> metricVector;

    const MCParticleList *pMCParticleList = nullptr;
    StatusCode mcReturn = PandoraContentApi::GetList(*this, m_mcParticleListName, pMCParticleList);

    int toolNum = 0;
    if (printMetrics)
        this->setupMetricsPlot();

    for (ProtoHitVectorMap::value_type protoHitVectorPair : allProtoHitVectors)
    {
        if (protoHitVectorPair.second.size() == 0)
            continue;

        CartesianPointVector pointVector;
        CaloHitVector twoDHits;
        CartesianPointVector pointVectorMC;

        for (const auto &nextPoint : protoHitVectorPair.second)
        {
            pointVector.push_back(nextPoint.GetPosition3D());
            twoDHits.push_back(nextPoint.GetParentCaloHit2D());
        }

        const LArTPC *const pFirstLArTPC(this->GetPandora().GetGeometry()->GetLArTPCMap().begin()->second);
        metricParams params;

        params.layerPitch = pFirstLArTPC->GetWirePitchW();
        params.slidingFitWidth = m_slidingFitHalfWindow;

        if (mcReturn == STATUS_CODE_SUCCESS)
        {
            MCParticleList mcList(pMCParticleList->begin(), pMCParticleList->end());
            const MCParticle *const pMCParticle = LArMCParticleHelper::GetMainMCParticle(pPfo);
            const LArMCParticle *const pLArMCParticle(dynamic_cast<const LArMCParticle *>(pMCParticle));

            if (pLArMCParticle != NULL)
            {
                for (const auto &nextMCHit : pLArMCParticle->GetMCStepPositions())
                    pointVectorMC.push_back(LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit));
            }
        }

        threeDMetric metrics;
        this->initMetrics(metrics);
        LArMetricHelper::GetThreeDMetrics(this->GetPandora(), pPfo, pointVector, twoDHits, metrics, params, pointVectorMC);
        metrics.particleId = toolNum;
        if (printMetrics)
            this->plotMetrics(pPfo, metrics);
        ++toolNum;

        metricVector.push_back(std::make_pair(protoHitVectorPair.first, metrics));
    }

    if (metricVector.size() == 0)
        return;

    if (printMetrics && metricVector.size() == 0) {
        this->tearDownMetricsPlot(false);
        return;
    } else if (printMetrics) {
        this->tearDownMetricsPlot(true);
    }

    if (!printMetrics && visualiseHits)
        this->PlotProjectedHits(metricVector, allProtoHitVectors);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::OutputCSVs(
        const ParticleFlowObject *const pPfo,
        const ProtoHitVectorMap &allProtoHitVectors,
        const std::vector<std::pair<std::string, ProtoHitVector>> &allProtoHitsToPlot,
        const std::vector<std::pair<std::string, ParameterVector>> &parameterVectors
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

    for (auto v : parameterVectors)
    {
        auto name = v.first;
        auto inliers = v.second;

        if (inliers.size() > 0)
        {
            csvFile << "X, Y, Z, ChiSquared, Interpolated, ToolName" << std::endl;

            for (auto &inlier : inliers)
            {
                auto hit = *std::dynamic_pointer_cast<Point3D>(inlier);
                csvFile << hit[0] << ","
                    << hit[1] << ","
                    << hit[2] << ","
                    << hit.m_ProtoHit.GetChi2() << ","
                    << (hit.m_ProtoHit.IsInterpolated() ? 1 : 0) << ","
                    << name
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

void ThreeDHitCreationAlgorithm::PlotProjectedHits(
        const std::vector<std::pair<std::string, threeDMetric>> &metricVector,
        const ProtoHitVectorMap &allProtoHitVectors
) const
{
    bool visualise2DHits = false;
    bool visualiseCaloHits = false;
    bool visualise3DHits = true;

    std::vector<Color> colours = {GREEN, RED, TEAL, GRAY, DARKRED, DARKGREEN, DARKPINK};
    std::vector<HitType> views = {TPC_VIEW_U, TPC_VIEW_V, TPC_VIEW_W};
    std::vector<std::string> viewNames = {"U", "V", "W"};
    CartesianPointVector actualTwoDHits;
    std::cout << "Need to draw " << metricVector.size() << " outputs..." << std::endl;
    int col = 0;

    for (unsigned int m = 0; m < metricVector.size(); ++m) {

        auto toolName = metricVector[m].first;
        auto protoVec = allProtoHitVectors.at(toolName);
        std::cout << "  Drawing Tool " << toolName << "." << std::endl;

        // Get the final hits and plot them out.
        int count = 0;
        int countWithCut = 0;

        for (unsigned int i = 0; i < protoVec.size(); ++i) {
            auto hit = protoVec[i];
            auto view = hit.GetParentCaloHit2D()->GetHitType();

            if (view != TPC_VIEW_W)
                continue;

            CartesianVector projHit = LArGeometryHelper::ProjectPosition(
                this->GetPandora(), hit.GetPosition3D(), view
            );
            CartesianVector twoDHit = hit.GetParentCaloHit2D()->GetPositionVector();

            Color hitColour;

            float mag = std::abs(projHit.GetX() - twoDHit.GetX()) + std::abs(projHit.GetZ() - twoDHit.GetZ());
            std::cout << "Mag: " << mag << std::endl;

            if (mag < 2)
                ++countWithCut;

            hitColour = colours[col];
            ++count;

            if (visualise2DHits)
                PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &projHit, "projected3DHits_" + toolName + "_" + std::to_string(mag), hitColour, 1));

            if (visualise3DHits)
                PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &hit.GetPosition3D(), "3DHits_" + toolName + "_" + std::to_string(mag), hitColour, 1));

            if (std::find(actualTwoDHits.begin(), actualTwoDHits.end(), twoDHit) == actualTwoDHits.end())
                actualTwoDHits.push_back(twoDHit);
        }

        ++col;

        std::cout << "    It had " << count << " hits to draw." << std::endl;
        if (countWithCut < count)
            std::cout << "    Could have " << countWithCut << " hits to draw, if using cut." << std::endl;
    }

    if (visualiseCaloHits)
        for (auto hit : actualTwoDHits)
            PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &hit, "actual2DHits", BLUE, 1));

}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::InterpolationMethod(const ParticleFlowObject *const pfo, ProtoHitVector &protoHitVector) const
{
    // If there is no hits at all....we can't do any interpolation.
    if (protoHitVector.empty())
        return;

    // Get the list of remaining hits for the current PFO.
    // That is, the 2D hits that do not have an associated 3D hit.
    CaloHitVector remainingTwoDHits;
    this->SeparateTwoDHits(pfo, protoHitVector, remainingTwoDHits);

    // If there is no remaining hits, then we don't need to interpolate anything.
    if (remainingTwoDHits.empty())
        return;

    // Get the current sliding linear fit, such that we can produce a point
    // that fits on to that fit.
    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(100); // TODO: Check if this should be the same or different.

    // Lets store the chi2 so that we can check against it later.
    double originalChi2(0.);
    CartesianPointVector currentPoints3D;
    this->ExtractResults(protoHitVector, originalChi2, currentPoints3D);

    if (currentPoints3D.size() <= 1)
        return;

    const ThreeDSlidingFitResult slidingFitResult(&currentPoints3D, layerWindow, layerPitch);
    // CartesianVector fitDirection = slidingFitResult.GetGlobalMaxLayerDirection();

    float managedToSet = 0;

    // We can then look over all these remaining hits and interpolate them.
    // For each hit, we want to compare it to the sliding linear fit, get the
    // points near by to this one and then interpolate the 3D hit from there,
    // using the linked 3D hit from the close by 2D hits that do have a
    // produced 3D hit.
    for (const pandora::CaloHit* currentCaloHit : remainingTwoDHits)
    {
        const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(currentCaloHit);

        // Get the position relative to the fit for the point.
        const float rL(slidingFitResult.GetLongitudinalDisplacement(pointPosition));

        // Attempt to interpolate the 2D hit.
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

        // Project the hit into 2D and get the distance between the projected
        // interpolated hit, and the original 2D hit.
        CartesianVector projectedHit = LArGeometryHelper::ProjectPosition(
                this->GetPandora(),
                projectedPosition,
                currentCaloHit->GetHitType()
        );
        double distanceBetweenHitsSqrd = (
                (currentCaloHit->GetPositionVector() - projectedHit).GetMagnitudeSquared()
        );

        // Using this distance, calculate a chi2 value for the interpolated hit.
        const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
        const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);
        double interpolatedChi2 = (distanceBetweenHitsSqrd) / (sigma3DFit * sigma3DFit);

        // Set the interpolated hit to have the calculated 3D position, with the chi2.
        interpolatedHit.SetPosition3D(projectedPosition, interpolatedChi2, true);
        interpolatedHit.AddTrajectorySample(
                TrajectorySample(projectedPosition, currentCaloHit->GetHitType(), sigmaUVW)
        );

        // Add the interpolated hit to the protoHitVector.
        protoHitVector.push_back(interpolatedHit);

        CartesianVector caloHit3D(
                currentCaloHit->GetPositionVector().GetX(),
                projectedPosition.GetY(),
                currentCaloHit->GetPositionVector().GetZ()
        );
        ++managedToSet;
    }

    // If we've interpolated at least 80% of this particle, we shouldn't
    // really be using it.
    //
    // TODO: Swap to option.
    // TODO: This ideally would be earlier on, and wouldn't clear, but just drop the interpolated.
    if (managedToSet >= (0.8 * protoHitVector.size()))
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
            std::cout << "ThreeDHitCreationAlgorithm::IterativeTreatment - Unexpected number of trajectory samples" << std::endl;
            throw StatusCodeException(STATUS_CODE_FAILURE);
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

void ThreeDHitCreationAlgorithm::initMetrics(threeDMetric &metricStruct) {
    // Set everything to -999, so we know it failed.
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

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MCParticleListName", m_mcParticleListName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "TrackMVAFileName", m_trackMVAFileName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MetricTreeFileName", m_metricFileName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MetricTreeName", m_metricTreeName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterateTrackHits", m_iterateTrackHits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterateShowerHits", m_iterateShowerHits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UseInterpolation", m_useInterpolation));

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
