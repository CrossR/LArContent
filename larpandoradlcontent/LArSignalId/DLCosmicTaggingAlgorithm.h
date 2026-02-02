/**
 *  @file   larpandoradlcontent/LArSignalId/DLCosmicTaggingAlgorithm.h
 *
 *  @brief  Header file for the deep learning slice hit tagging.
 *
 *  $Log: $
 */
#include "larpandoradlcontent/LArVertex/DlVertexingBaseAlgorithm.h"
#ifndef LAR_DL_COSMIC_TAGGING_ALGORITHM_H
#define LAR_DL_COSMIC_TAGGING_ALGORITHM_H 1

#include "Pandora/Algorithm.h"
#include "Pandora/AlgorithmHeaders.h"

#include "larpandoradlcontent/LArVertex/DlVertexingBaseAlgorithm.h"

using namespace lar_content;

namespace lar_dl_content
{
/**
 *  @brief  DLCosmicTaggingAlgorithm class
 */
class DLCosmicTaggingAlgorithm : public DlVertexingBaseAlgorithm
{
public:
    typedef std::map<const pandora::CaloHit *, std::tuple<int, int>> CaloHitToPixelMap;

    /**
     *  @brief Default constructor
     */
    DLCosmicTaggingAlgorithm();

    virtual ~DLCosmicTaggingAlgorithm();

protected:

    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
    pandora::StatusCode PrepareTrainingSample();
    pandora::StatusCode Infer();

    /**
     *  @brief  Populate a root true with vertex information.
     */
    void PopulateRootTree() const;

private:

    /*
     *  @brief  Create input for the network from a calo hit list
     *
     *  @param  caloHits The CaloHitList from which the input should be made
     *  @param  view The wire plane view
     *  @param  xMin The minimum x coordinate for the hits
     *  @param  xMax The maximum x coordinate for the hits
     *  @param  zMin The minimum x coordinate for the hits
     *  @param  zMax The maximum x coordinate for the hits
     *  @param  networkInput The TorchInput object to populate
     *  @param  pixelVector The output vector of populated pixels
     *  @param  caloHitToPixelMap The map from calo hits to pixel positions
     *
     *  @return The StatusCode resulting from the function
     **/
    pandora::StatusCode MakeNetworkInputFromHits(const pandora::CaloHitList &caloHits, const pandora::HitType view, const float xMin,
        const float xMax, const float zMin, const float zMax, LArDLHelper::TorchInput &networkInput, PixelVector &pixelVector,
        CaloHitToPixelMap &caloHitToPixelMap) const;

    int m_event;                ///< The current event number
    bool m_visualise;           ///< Whether or not to visualise the candidate vertices
    bool m_writeTree;           ///< Whether or not to write validation details to a ROOT tree
    std::string m_rootTreeName; ///< The ROOT tree name
    std::string m_rootFileName; ///< The ROOT file name
    std::mt19937 m_rng;         ///< The random number generator
};

} // namespace lar_dl_content

#endif // LAR_DL_COSMIC_TAGGING_ALGORITHM_H
