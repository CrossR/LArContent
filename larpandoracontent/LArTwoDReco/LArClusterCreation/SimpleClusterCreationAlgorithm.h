/**
 *  @file   larpandoracontent/LArTwoDReco/LArClusterCreation/SimpleClusterCreationAlgorithm.h
 *
 *  @brief  Header file for the cluster creation algorithm class.
 *
 *  $Log: $
 */
#ifndef LAR_SIMPLE_CLUSTER_CREATION_ALGORITHM_H
#define LAR_SIMPLE_CLUSTER_CREATION_ALGORITHM_H 1

#include "Pandora/Algorithm.h"

#include "larpandoracontent/LArUtility/KDTreeLinkerAlgoT.h"

#include <string>
#include <unordered_map>

namespace lar_content
{

/**
 *  @brief  SimpleClusterCreationAlgorithm class
 */
class SimpleClusterCreationAlgorithm : public pandora::Algorithm
{
public:
    /**
     *  @brief  Default constructor
     */
    SimpleClusterCreationAlgorithm();

protected:
    typedef std::unordered_map<const pandora::CaloHit *, pandora::CaloHitVector> HitAssociationMap;

    /**
     *  @brief  Find the hits within the clustering window of each hit
     *
     *  @param  caloHitVector the calo hits
     *  @param  hitAssociationMap to receive, for each hit with any, its nearby hits sorted by position
     */
    virtual void BuildAssociationMap(const pandora::CaloHitVector &caloHitVector, HitAssociationMap &hitAssociationMap) const;

    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    float m_clusteringWindowSquared;     ///< Maximum distance (squared) for two hits to be joined
    std::string m_inputCaloHitListName;  ///< Name of the input calo hit list (empty: use the current list)
    std::string m_outputClusterListName; ///< Name to save the new clusters under (empty: leave in the current cluster list)

private:
    pandora::StatusCode Run();

    /**
     *  @brief  Create a cluster from each group of hits joined together by the associations
     *
     *  @param  caloHitVector the calo hits, sorted by position
     *  @param  hitAssociationMap the associations between hits
     */
    void CreateClusters(const pandora::CaloHitVector &caloHitVector, const HitAssociationMap &hitAssociationMap) const;
};

} // namespace lar_content

#endif // #ifndef LAR_SIMPLE_CLUSTER_CREATION_ALGORITHM_H
